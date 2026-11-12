"""
AIRCONTROL — CAMERA MODULE (STEP 2: THREAD ARCHITECTURE SEPARATION)
Separates camera capture (Camera Worker) and hand/gesture inference (Vision Worker)
into dedicated background threads communicating via bounded lock-free queues.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from config import (
    CAMERA_HEIGHT,
    CAMERA_QUEUE_MAXSIZE,
    CAMERA_WIDTH,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_MIRROR_FEED,
    TARGET_FPS,
    VISION_QUEUE_MAXSIZE,
    WORKER_SHUTDOWN_TIMEOUT,
)
from gestures.gesture_engine import GestureEngine, GestureTelemetry
from utils.diagnostics import measure_time
from utils.fps import FPSCounter
from vision.hand_tracker import HandTracker, HandTrackingResult

logger = logging.getLogger("AirControl.Camera")


@dataclass
class FrameData:
    """Thread-safe container passed to the UI layer on each video tick."""
    display_frame: np.ndarray              # BGR frame annotated for UI rendering
    raw_frame: np.ndarray                  # Raw mirrored frame (unannotated)
    tracking_result: HandTrackingResult    # Extracted hand landmarks & telemetry
    fps: float                             # Smoothed FPS
    latency_ms: float                      # Inter-frame latency in ms
    resolution: Tuple[int, int]            # (width, height)
    camera_status: str                     # "ONLINE", "OFFLINE", "ERROR", etc.
    camera_index: int
    air_mouse_telemetry: Optional[object] = None  # Live Air Mouse telemetry snapshot
    gesture_telemetry: Optional[GestureTelemetry] = None  # Stage 3 Gesture Intelligence telemetry


class CameraManager:
    """
    Manages OpenCV webcam capture and vision inference across two dedicated background workers:
    1. CAMERA WORKER: Frame capture from hardware sensor only.
    2. VISION WORKER: MediaPipe landmark extraction, gesture engine & air mouse.
    Guarantees that UI thread operations never stall or block.
    """

    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps_target: int = TARGET_FPS,
        mirror_feed: bool = DEFAULT_MIRROR_FEED,
        hand_tracker: Optional[HandTracker] = None,
        air_mouse: Optional[object] = None,
        gesture_engine: Optional[GestureEngine] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        self._camera_index = camera_index
        self._target_width = width
        self._target_height = height
        self._target_fps = fps_target
        self._mirror_feed = mirror_feed

        self.hand_tracker = hand_tracker or HandTracker()
        self.air_mouse = air_mouse
        self.gesture_engine = gesture_engine or GestureEngine()
        self.on_status_change = on_status_change

        self._fps_counter = FPSCounter(window_size=30)

        # Bounded Queues
        # 1. Camera Worker -> Vision Worker
        self._raw_queue: queue.Queue[Tuple[np.ndarray, int, int]] = queue.Queue(maxsize=CAMERA_QUEUE_MAXSIZE)
        # 2. Vision Worker -> Main UI Thread
        self._frame_queue: queue.Queue[FrameData] = queue.Queue(maxsize=VISION_QUEUE_MAXSIZE)

        # Worker Threads
        self._camera_thread: Optional[threading.Thread] = None
        self._vision_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._status = "OFFLINE"
        self._draw_skeleton = True
        self._draw_hud = True
        self._actual_resolution: Tuple[int, int] = (width, height)

    @property
    def status(self) -> str:
        """Current operational status."""
        with self._lock:
            return self._status

    @property
    def is_running(self) -> bool:
        """True if worker threads are alive and running."""
        cam_alive = self._camera_thread is not None and self._camera_thread.is_alive()
        vis_alive = self._vision_thread is not None and self._vision_thread.is_alive()
        return cam_alive or vis_alive

    @property
    def camera_index(self) -> int:
        with self._lock:
            return self._camera_index

    @property
    def mirror_feed(self) -> bool:
        with self._lock:
            return self._mirror_feed

    @mirror_feed.setter
    def mirror_feed(self, value: bool) -> None:
        with self._lock:
            self._mirror_feed = value

    @property
    def draw_skeleton(self) -> bool:
        with self._lock:
            return self._draw_skeleton

    @draw_skeleton.setter
    def draw_skeleton(self, value: bool) -> None:
        with self._lock:
            self._draw_skeleton = value

    @property
    def draw_hud(self) -> bool:
        with self._lock:
            return self._draw_hud

    @draw_hud.setter
    def draw_hud(self, value: bool) -> None:
        with self._lock:
            self._draw_hud = value

    def _set_status(self, new_status: str) -> None:
        """Internal helper to update status and notify listeners."""
        with self._lock:
            self._status = new_status
        logger.info("Camera status updated: %s", new_status)
        if self.on_status_change:
            try:
                self.on_status_change(new_status)
            except Exception as e:
                logger.error("Error in on_status_change callback: %s", e)

    def start(self, camera_index: Optional[int] = None) -> bool:
        """Start both Camera Worker and Vision Worker background threads."""
        with self._lock:
            if camera_index is not None:
                self._camera_index = camera_index

            if self.is_running:
                logger.warning("CameraManager is already running.")
                return True

            self._stop_event.clear()
            self._fps_counter.reset()
            self._set_status("STARTING")

            # 1. Spawn Camera Worker
            self._camera_thread = threading.Thread(
                target=self._camera_worker,
                name="AirControl-CameraWorker",
                daemon=True,
            )
            # 2. Spawn Vision Worker
            self._vision_thread = threading.Thread(
                target=self._vision_worker,
                name="AirControl-VisionWorker",
                daemon=True,
            )

            self._camera_thread.start()
            self._vision_thread.start()
            return True

    def stop(self, timeout: float = WORKER_SHUTDOWN_TIMEOUT) -> None:
        """Signal worker threads to terminate and safely release hardware."""
        with self._lock:
            if not self.is_running and self._status == "OFFLINE":
                return

            cam_thread = self._camera_thread
            vis_thread = self._vision_thread
            cap_to_release = self._cap
            self._camera_thread = None
            self._vision_thread = None
            self._cap = None
            self._stop_event.set()

        logger.info("Stopping camera and vision worker threads...")
        if cap_to_release is not None:
            try:
                cap_to_release.release()
            except Exception as e:
                logger.debug("Exception releasing cap early: %s", e)

        # Join workers with bounded non-blocking timeout
        if cam_thread is not None and cam_thread.is_alive():
            cam_thread.join(timeout=timeout)
            if cam_thread.is_alive():
                logger.warning("Camera thread did not exit within timeout.")

        if vis_thread is not None and vis_thread.is_alive():
            vis_thread.join(timeout=timeout)
            if vis_thread.is_alive():
                logger.warning("Vision thread did not exit within timeout.")

        self._set_status("OFFLINE")

        # Clear residual queue contents
        while not self._raw_queue.empty():
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                break

        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def restart(self, new_camera_index: int) -> bool:
        """Switch camera sources cleanly."""
        self.stop()
        return self.start(camera_index=new_camera_index)

    def get_latest_frame(self, block: bool = False, timeout: Optional[float] = None) -> Optional[FrameData]:
        """
        Retrieve the latest frame from the thread-safe queue.
        Drains older queued items to ensure zero display latency.
        """
        latest_item = None
        try:
            if block:
                latest_item = self._frame_queue.get(block=True, timeout=timeout)
            while True:
                latest_item = self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        return latest_item

    def _open_capture(self, index: int) -> Optional[cv2.VideoCapture]:
        """
        Attempt opening camera device using DirectShow on Windows with verified frame read.
        If requested index fails, automatically probes indices 0, 1, 2.
        """
        indices_to_try = [index] + [i for i in [0, 1, 2] if i != index]

        for idx in indices_to_try:
            logger.info("Probing camera index %d with cv2.CAP_DSHOW...", idx)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    with self._lock:
                        self._camera_index = idx

                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
                        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
                    except Exception as e:
                        logger.warning("Could not set camera properties on index %d: %s", idx, e)

                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._actual_resolution = (actual_w, actual_h)
                    logger.info("Camera %d verified and opened successfully (%dx%d).", idx, actual_w, actual_h)
                    return cap
                else:
                    logger.warning("Camera %d opened with DSHOW but failed initial frame read.", idx)
                    cap.release()
            else:
                cap.release()

        # Fallback to default backend
        for idx in indices_to_try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    with self._lock:
                        self._camera_index = idx
                    self._actual_resolution = (
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    )
                    logger.info("Camera %d opened via default backend.", idx)
                    return cap
                cap.release()

        logger.error("No working camera found on any probed indices.")
        return None

    # ==========================================================================
    # WORKER 1: CAMERA CAPTURE WORKER
    # ==========================================================================
    def _camera_worker(self) -> None:
        """
        Dedicated camera capture thread: runs OpenCV VideoCapture read loop.
        Produces raw frames and pushes to _raw_queue without running inference.
        """
        cap = None
        try:
            with self._lock:
                cam_idx = self._camera_index

            cap = self._open_capture(cam_idx)
            if cap is None:
                self._set_status("ERROR_NO_CAMERA")
                return

            with self._lock:
                self._cap = cap

            self._set_status("ONLINE")

            failed_reads = 0
            max_failed_reads = 30

            while not self._stop_event.is_set():
                with measure_time("CAMERA_CAPTURE", threshold_ms=40.0):
                    ret, frame = cap.read()

                if not ret or frame is None or frame.size == 0:
                    failed_reads += 1
                    logger.warning("Camera frame read failed (%d/%d)", failed_reads, max_failed_reads)
                    if failed_reads >= max_failed_reads:
                        logger.error("Exceeded maximum consecutive read failures. Camera disconnected.")
                        self._set_status("DISCONNECTED")
                        break
                    time.sleep(0.03)
                    continue

                failed_reads = 0
                timestamp_ms = int(time.time() * 1000)

                # Push raw frame to Vision Worker queue (Drop oldest strategy)
                if self._raw_queue.full():
                    try:
                        self._raw_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self._raw_queue.put_nowait((frame, timestamp_ms, cam_idx))
                except queue.Full:
                    pass

                time.sleep(0.001)

        except Exception as e:
            logger.exception("Unexpected exception in camera capture loop: %s", e)
            self._set_status("ERROR")
        finally:
            if cap is not None:
                try:
                    cap.release()
                    logger.info("OpenCV VideoCapture released cleanly.")
                except Exception as e:
                    logger.warning("Exception releasing VideoCapture: %s", e)
            if self._status not in ("ERROR", "ERROR_NO_CAMERA", "ERROR_MODEL", "DISCONNECTED"):
                self._set_status("OFFLINE")
            logger.info("Camera capture worker thread exited.")

    # ==========================================================================
    # WORKER 2: VISION INFERENCE WORKER
    # ==========================================================================
    def _vision_worker(self) -> None:
        """
        Dedicated vision worker: pulls raw frames, runs MediaPipe HandTracker,
        evaluates gestures, updates Air Mouse, renders HUD, and pushes FrameData to UI.
        """
        try:
            if not self.hand_tracker.initialize():
                self._set_status("ERROR_MODEL")
                logger.error("Hand tracker initialization failed in vision thread.")
                return

            while not self._stop_event.is_set():
                try:
                    raw_item = self._raw_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if self._stop_event.is_set():
                    break

                # Skip outdated frames in queue to maintain zero display latency
                while not self._raw_queue.empty():
                    try:
                        raw_item = self._raw_queue.get_nowait()
                    except queue.Empty:
                        break

                raw_frame, timestamp_ms, cam_idx = raw_item

                with self._lock:
                    is_mirrored = self._mirror_feed
                    should_draw_skeleton = self._draw_skeleton
                    should_draw_hud = self._draw_hud

                if is_mirrored:
                    frame = cv2.flip(raw_frame, 1)
                else:
                    frame = raw_frame

                mirrored_raw = frame.copy()

                # 1. MediaPipe Hand Tracking
                with measure_time("HAND_TRACKING", threshold_ms=40.0):
                    tracking_result = self.hand_tracker.process_frame(
                        bgr_frame=frame,
                        timestamp_ms=timestamp_ms,
                        is_mirrored=is_mirrored,
                    )

                # 2. Stage 3 Gesture Intelligence
                gesture_telemetry = None
                if self.gesture_engine is not None:
                    try:
                        with measure_time("GESTURE_EVALUATION", threshold_ms=15.0):
                            gesture_telemetry = self.gesture_engine.process(
                                tracking_result=tracking_result,
                                timestamp_ms=timestamp_ms,
                            )
                    except Exception as e:
                        logger.debug("GestureEngine process exception: %s", e)

                # 3. Stage 2 Relative Air Mouse
                air_mouse_telemetry = None
                if self.air_mouse is not None:
                    try:
                        with measure_time("AIR_MOUSE", threshold_ms=10.0):
                            air_mouse_telemetry = self.air_mouse.process_frame(tracking_result)
                    except Exception as e:
                        logger.debug("AirMouse process_frame exception: %s", e)

                # 4. HUD & Skeleton Rendering
                annotated_frame = frame
                if should_draw_skeleton and tracking_result.hands:
                    annotated_frame = self.hand_tracker.draw_landmarks(
                        frame=frame,
                        result=tracking_result,
                        draw_skeleton=should_draw_skeleton,
                        draw_hud=should_draw_hud,
                        air_mouse_telemetry=air_mouse_telemetry,
                        gesture_telemetry=gesture_telemetry,
                    )

                fps = self._fps_counter.tick()
                latency_ms = self._fps_counter.latency_ms

                frame_data = FrameData(
                    display_frame=annotated_frame,
                    raw_frame=mirrored_raw,
                    tracking_result=tracking_result,
                    fps=fps,
                    latency_ms=latency_ms,
                    resolution=self._actual_resolution,
                    camera_status=self.status,
                    camera_index=cam_idx,
                    air_mouse_telemetry=air_mouse_telemetry,
                    gesture_telemetry=gesture_telemetry,
                )

                # Push to UI queue (Drop oldest strategy)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self._frame_queue.put_nowait(frame_data)
                except queue.Full:
                    pass

        except Exception as e:
            logger.exception("Unexpected exception in vision worker loop: %s", e)
        finally:
            logger.info("Vision worker thread exited.")
