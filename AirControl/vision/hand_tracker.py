"""
AIRCONTROL — HAND TRACKER MODULE
Implements Google MediaPipe Hand Landmarker Tasks API in VIDEO mode.
Detects up to 2 hands, extracts 21 3D landmarks, computes handedness, and renders
a futuristic sci-fi skeleton HUD with high-contrast glowing elements.
"""

import logging
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand3d import Hand3DProcessor, HandModel3D
from config import (
    COLOR_CYAN,
    COLOR_PURPLE,
    FINGERTIP_INDICES,
    HAND_CONNECTIONS,
    LANDMARK_NAMES,
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    MODEL_URL,
)

logger = logging.getLogger("AirControl.HandTracker")


@dataclass
class HandData:
    """Structured telemetry data for a single detected hand."""
    index: int
    handedness: str                       # "Left" or "Right"
    confidence: float                     # 0.0 - 1.0
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) normalized [0, 1]
    landmarks_pixel: List[Tuple[int, int]]             # (px, py) image coordinates
    fingertips_pixel: Dict[str, Tuple[int, int]]       # Fingertip name -> (px, py)
    palm_center: Tuple[int, int]                       # Palm centroid (px, py)
    bounding_box: Tuple[int, int, int, int]            # (min_x, min_y, max_x, max_y)
    finger_states: Dict[str, bool] = field(default_factory=dict)  # Finger name -> is_extended
    landmarks_world: List[Tuple[float, float, float]] = field(default_factory=list)  # 3D metric world coordinates
    hand_model_3d: Optional[HandModel3D] = None        # Articulated 3D skeletal model


@dataclass
class HandTrackingResult:
    """Consolidated tracking result for a processed frame."""
    hands: List[HandData] = field(default_factory=list)
    num_hands: int = 0
    inference_time_ms: float = 0.0
    is_success: bool = True
    error_message: Optional[str] = None


class HandTracker:
    """Encapsulates MediaPipe Hand Landmarker Tasks API for video stream tracking."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        model_url: str = MODEL_URL,
        max_hands: int = MAX_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_presence_confidence: float = MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ):
        self.model_path = Path(model_path)
        self.model_url = model_url
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._landmarker: Optional[vision.HandLandmarker] = None
        self._last_timestamp_ms: int = -1
        self._is_initialized = False

        # Articulated 3D Hand Processors
        self._processor_right = Hand3DProcessor("Right")
        self._processor_left = Hand3DProcessor("Left")

        # Staged Diagnostics Telemetry
        self._diag_frame_count: int = 0
        self._last_diag_detected_count: int = -1

        # Pre-convert hex colors to OpenCV BGR format
        self._bgr_cyan = self._hex_to_bgr(COLOR_CYAN)      # Primary Right Hand
        self._bgr_purple = self._hex_to_bgr(COLOR_PURPLE)  # Primary Left Hand
        self._bgr_white = (255, 255, 255)
        self._bgr_black = (10, 10, 10)
        self._bgr_accent_tip = (255, 128, 0)               # Bright amber/yellow
        self._bgr_glow_pink = (200, 0, 255)                # Magenta accent

    @staticmethod
    def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
        """Convert a '#RRGGBB' hex color to OpenCV (B, G, R) tuple."""
        hex_str = hex_str.lstrip("#")
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (b, g, r)

    def ensure_model_available(self) -> bool:
        """Verify the model file exists and is valid; download if necessary."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            if self.model_path.exists():
                file_size = self.model_path.stat().st_size
                # Valid Hand Landmarker model is ~7.8MB (> 1MB check)
                if file_size > 1_000_000:
                    logger.info("Found valid MediaPipe model at %s (%d bytes)", self.model_path, file_size)
                    return True
                else:
                    logger.warning("Existing model file is undersized (%d bytes). Re-downloading...", file_size)
                    self.model_path.unlink(missing_ok=True)

            logger.info("Downloading official Hand Landmarker model from: %s", self.model_url)
            urllib.request.urlretrieve(self.model_url, str(self.model_path))

            if self.model_path.exists() and self.model_path.stat().st_size > 1_000_000:
                logger.info("Model download complete. Saved to: %s", self.model_path)
                return True
            else:
                logger.error("Downloaded model file failed size validation.")
                return False

        except Exception as e:
            logger.error("Failed to download or verify MediaPipe model: %s", e)
            return False

    def initialize(self) -> bool:
        """Initialize the MediaPipe Hand Landmarker Tasks API in VIDEO mode."""
        if self._is_initialized and self._landmarker is not None:
            return True

        if not self.ensure_model_available():
            logger.error("Cannot initialize HandTracker: model file not available.")
            return False

        try:
            logger.info("Initializing MediaPipe HandLandmarker Tasks API (VIDEO mode)...")
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._is_initialized = True
            self._last_timestamp_ms = -1
            logger.info("MediaPipe HandLandmarker Tasks API initialized successfully.")
            return True

        except Exception as e:
            logger.exception("MediaPipe HandLandmarker initialization failed: %s", e)
            self._is_initialized = False
            self._landmarker = None
            return False

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: int,
        is_mirrored: bool = True,
    ) -> HandTrackingResult:
        """
        Process a BGR frame through the Hand Landmarker Tasks API.

        Args:
            bgr_frame: OpenCV BGR image array.
            timestamp_ms: Current timestamp in milliseconds (monotonically increasing).
            is_mirrored: True if the frame was flipped horizontally for mirror/selfie mode.

        Returns:
            HandTrackingResult containing extracted hands, landmarks, and telemetry.
        """
        if not self._is_initialized or self._landmarker is None:
            if not self.initialize():
                return HandTrackingResult(
                    is_success=False,
                    error_message="Hand Landmarker is not initialized.",
                )

        # Enforce strictly monotonic timestamps for MediaPipe VIDEO mode
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        t0 = time.perf_counter()
        h, w = bgr_frame.shape[:2]
        c = bgr_frame.shape[2] if len(bgr_frame.shape) > 2 else 1

        self._diag_frame_count += 1
        frame_id = self._diag_frame_count

        try:
            # 1. Convert BGR to RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # 2. Wrap into MediaPipe Image container
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 3. Perform Video Inference
            t_infer_start = time.perf_counter()
            detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            inference_ms = (time.perf_counter() - t_infer_start) * 1000.0

            # 4. Extract structured telemetry
            hands_list: List[HandData] = []
            raw_landmarks = detection_result.hand_landmarks
            raw_handedness = detection_result.handedness
            raw_world = getattr(detection_result, "hand_world_landmarks", None)

            num_2d_hands = len(raw_landmarks) if raw_landmarks else 0
            num_world_hands = len(raw_world) if raw_world else 0
            num_handedness = len(raw_handedness) if raw_handedness else 0
            diag_3d_success = 0
            diag_3d_failed = 0

            if raw_landmarks:
                for idx, landmarks in enumerate(raw_landmarks):
                    try:
                        # Extract Handedness and confidence
                        hand_label = "Unknown"
                        confidence = 0.0

                        if raw_handedness and idx < len(raw_handedness) and raw_handedness[idx]:
                            category = raw_handedness[idx][0]
                            orig_label = category.category_name or category.display_name or "Unknown"
                            confidence = float(category.score)

                            # When the webcam image is mirrored (selfie mode), MediaPipe's
                            # internal projection inverts left/right. We correct it to match
                            # the user's actual anatomical hand:
                            if is_mirrored:
                                hand_label = "Right" if orig_label == "Left" else "Left"
                            else:
                                hand_label = orig_label

                        # Extract normalized & pixel coordinates
                        norm_coords: List[Tuple[float, float, float]] = []
                        pixel_coords: List[Tuple[int, int]] = []
                        xs, ys = [], []

                        for lm in landmarks:
                            norm_coords.append((float(lm.x), float(lm.y), float(lm.z)))
                            px = int(np.clip(lm.x * w, 0, w - 1))
                            py = int(np.clip(lm.y * h, 0, h - 1))
                            pixel_coords.append((px, py))
                            xs.append(px)
                            ys.append(py)

                        # Fingertips (Thumb:4, Index:8, Middle:12, Ring:16, Pinky:20)
                        fingertips: Dict[str, Tuple[int, int]] = {
                            "Thumb": pixel_coords[4],
                            "Index": pixel_coords[8],
                            "Middle": pixel_coords[12],
                            "Ring": pixel_coords[16],
                            "Pinky": pixel_coords[20],
                        }

                        # Palm Center (Centroid of Wrist and MCPs: 0, 5, 9, 13, 17)
                        palm_indices = (0, 5, 9, 13, 17)
                        palm_x = int(sum(pixel_coords[i][0] for i in palm_indices) / len(palm_indices))
                        palm_y = int(sum(pixel_coords[i][1] for i in palm_indices) / len(palm_indices))

                        # Bounding Box with padding
                        min_x, max_x = max(0, min(xs) - 15), min(w, max(xs) + 15)
                        min_y, max_y = max(0, min(ys) - 15), min(h, max(ys) + 15)

                        # Compute Finger Extension States (True = extended, False = curled)
                        wrist_pt = norm_coords[0]
                        def _dist_sq(p1, p2):
                            return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2

                        # Thumb extended if tip (4) is further from pinky MCP (17) than thumb IP (3)
                        thumb_tip_pinky = _dist_sq(norm_coords[4], norm_coords[17])
                        thumb_ip_pinky = _dist_sq(norm_coords[3], norm_coords[17])
                        thumb_ext = thumb_tip_pinky > thumb_ip_pinky * 1.15

                        finger_states: Dict[str, bool] = {
                            "Thumb": thumb_ext,
                            "Index": _dist_sq(norm_coords[8], wrist_pt) > _dist_sq(norm_coords[6], wrist_pt),
                            "Middle": _dist_sq(norm_coords[12], wrist_pt) > _dist_sq(norm_coords[10], wrist_pt),
                            "Ring": _dist_sq(norm_coords[16], wrist_pt) > _dist_sq(norm_coords[14], wrist_pt),
                            "Pinky": _dist_sq(norm_coords[20], wrist_pt) > _dist_sq(norm_coords[18], wrist_pt),
                        }

                        # 2D Hand Data is ALWAYS created and guaranteed first (Single Source of Truth)
                        hand_data = HandData(
                            index=idx,
                            handedness=hand_label,
                            confidence=confidence,
                            landmarks_norm=norm_coords,
                            landmarks_pixel=pixel_coords,
                            fingertips_pixel=fingertips,
                            palm_center=(palm_x, palm_y),
                            bounding_box=(min_x, min_y, max_x, max_y),
                            finger_states=finger_states,
                            landmarks_world=list(norm_coords),
                            hand_model_3d=None,
                        )
                        hands_list.append(hand_data)

                        # 3D Extraction & Articulated Skeletal Processing (Layered Enhancement)
                        try:
                            world_coords: List[Tuple[float, float, float]] = []
                            if raw_world and idx < len(raw_world) and raw_world[idx]:
                                for wlm in raw_world[idx]:
                                    wx = float(wlm.x) if math.isfinite(wlm.x) else 0.0
                                    wy = float(wlm.y) if math.isfinite(wlm.y) else 0.0
                                    wz = float(wlm.z) if math.isfinite(wlm.z) else 0.0
                                    world_coords.append((wx, wy, wz))
                            else:
                                world_coords = list(norm_coords)

                            hand_data.landmarks_world = world_coords

                            processor = self._processor_right if hand_label == "Right" else self._processor_left
                            hand_model = processor.update(
                                joints_3d=world_coords,
                                joints_norm=norm_coords,
                                confidence=confidence,
                                timestamp=timestamp_ms / 1000.0,
                            )
                            hand_data.hand_model_3d = hand_model
                            diag_3d_success += 1
                        except Exception as e_3d:
                            logger.warning("3D articulated conversion skipped for hand %d (%s): %s", idx, hand_label, e_3d)
                            hand_data.hand_model_3d = None
                            diag_3d_failed += 1

                    except Exception as e_hand:
                        logger.error("Error processing hand #%d: %s", idx, e_hand)

            # Staged Diagnostics: periodic logging every 30 frames or on hand transition
            should_log_diag = (frame_id % 30 == 1) or (num_2d_hands != self._last_diag_detected_count)
            if should_log_diag:
                self._last_diag_detected_count = num_2d_hands
                state_str = f"{len(hands_list)} active ({[h.handedness for h in hands_list]})" if hands_list else "NO_HAND"
                logger.info(
                    "[DIAG 1-10] Frame #%d: received (%dx%d, %dch) | Inference: %.1fms | "
                    "2D Hands: %d | World: %d | Handedness: %d | 3D: %d ok / %d err | State: %s",
                    frame_id, w, h, c, inference_ms,
                    num_2d_hands, num_world_hands, num_handedness,
                    diag_3d_success, diag_3d_failed, state_str,
                )

            return HandTrackingResult(
                hands=hands_list,
                num_hands=len(hands_list),
                inference_time_ms=(time.perf_counter() - t0) * 1000.0,
                is_success=True,
                error_message=None,
            )

        except Exception as e:
            logger.exception("Error during hand tracking inference at frame #%d: %s", frame_id, e)
            return HandTrackingResult(
                hands=[],
                num_hands=0,
                inference_time_ms=(time.perf_counter() - t0) * 1000.0,
                is_success=False,
                error_message=str(e),
            )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: HandTrackingResult,
        draw_skeleton: bool = True,
        draw_hud: bool = True,
        air_mouse_telemetry: Optional[object] = None,
        gesture_telemetry: Optional[object] = None,
    ) -> np.ndarray:
        """
        Draw high-visibility, futuristic sci-fi hand skeleton, joints, fingertips,
        and HUD telemetry directly onto the frame.
        """
        if not result.hands or not draw_skeleton:
            return frame

        canvas = frame  # Draw in-place on frame

        for hand in result.hands:
            pts = hand.landmarks_pixel
            if len(pts) < 21:
                continue

            # Color scheme: Cyan for Right Hand, Purple/Violet for Left Hand
            is_right = hand.handedness == "Right"
            bone_color = self._bgr_cyan if is_right else self._bgr_purple
            accent_color = (255, 230, 0) if is_right else (255, 120, 255)

            # Check if this hand is the active Air Mouse controller
            is_mouse_hand = (
                air_mouse_telemetry is not None
                and getattr(air_mouse_telemetry, "enabled", False)
                and getattr(air_mouse_telemetry, "hand_used", None) == hand.handedness
            )

            # ------------------------------------------------------------------
            # 1. DRAW SKELETON CONNECTIONS (Dual-pass outline + neon glow)
            # ------------------------------------------------------------------
            # Pass 1: Dark high-contrast background stroke
            for p1, p2 in HAND_CONNECTIONS:
                pt1, pt2 = pts[p1], pts[p2]
                cv2.line(canvas, pt1, pt2, (15, 20, 30), thickness=4, lineType=cv2.LINE_AA)

            # Pass 2: Bright neon glowing bone line
            for p1, p2 in HAND_CONNECTIONS:
                pt1, pt2 = pts[p1], pts[p2]
                cv2.line(canvas, pt1, pt2, bone_color, thickness=2, lineType=cv2.LINE_AA)

            # ------------------------------------------------------------------
            # 2. DRAW PALM TARGET RETICLE
            # ------------------------------------------------------------------
            pc = hand.palm_center
            cv2.circle(canvas, pc, 10, (15, 20, 30), thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pc, 10, bone_color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pc, 2, self._bgr_white, thickness=-1, lineType=cv2.LINE_AA)

            # ------------------------------------------------------------------
            # 3. DRAW LANDMARK JOINTS (21 Points)
            # ------------------------------------------------------------------
            for idx, pt in enumerate(pts):
                if idx in FINGERTIP_INDICES:
                    # ----------------------------------------------------------
                    # 4. FINGERTIPS VISUALLY DISTINGUISHED
                    # Concentric glowing rings + bright contrasting center
                    # ----------------------------------------------------------
                    # Outer dark outline
                    cv2.circle(canvas, pt, 9, (10, 10, 10), thickness=3, lineType=cv2.LINE_AA)
                    # Outer neon ring
                    cv2.circle(canvas, pt, 8, accent_color, thickness=2, lineType=cv2.LINE_AA)
                    # Mid glow ring
                    cv2.circle(canvas, pt, 5, bone_color, thickness=1, lineType=cv2.LINE_AA)
                    # Inner solid dot
                    cv2.circle(canvas, pt, 3, self._bgr_white, thickness=-1, lineType=cv2.LINE_AA)
                else:
                    # General skeleton joint
                    cv2.circle(canvas, pt, 4, (10, 10, 10), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, pt, 3, bone_color, thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, pt, 1, self._bgr_white, thickness=-1, lineType=cv2.LINE_AA)

            # ------------------------------------------------------------------
            # 5. AIR MOUSE INTERACTION OVERLAYS
            # ------------------------------------------------------------------
            if is_mouse_hand:
                idx_tip = pts[8]
                thumb_tip = pts[4]

                # Air Mouse Cursor Reticle on Index Tip
                cv2.circle(canvas, idx_tip, 15, (0, 240, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.drawMarker(canvas, idx_tip, (0, 240, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

                # Visual Pinch Arc & Drag Link
                is_dragging = getattr(air_mouse_telemetry, "is_dragging", False)
                pinch_ratio = getattr(air_mouse_telemetry, "pinch_ratio", 1.0)
                if is_dragging or pinch_ratio < 0.25:
                    cv2.line(canvas, thumb_tip, idx_tip, (0, 255, 163), thickness=2, lineType=cv2.LINE_AA)
                    mid_pinch = ((thumb_tip[0] + idx_tip[0]) // 2, (thumb_tip[1] + idx_tip[1]) // 2)
                    cv2.circle(canvas, mid_pinch, 6, (0, 255, 163), thickness=-1, lineType=cv2.LINE_AA)

            # ------------------------------------------------------------------
            # 6. SCI-FI HUD BADGE (Handedness + Confidence score + Mouse State)
            # ------------------------------------------------------------------
            if draw_hud:
                min_x, min_y, max_x, _ = hand.bounding_box
                badge_x = max(10, min_x)
                badge_y = max(35, min_y - 12)

                text = f"{hand.handedness.upper()} [{int(hand.confidence * 100)}%]"
                if is_mouse_hand:
                    mouse_state = getattr(air_mouse_telemetry, "state", "ACTIVE")
                    text += f" | MOUSE: {mouse_state}"
                elif gesture_telemetry is not None and getattr(gesture_telemetry, "hands", None):
                    h_state = gesture_telemetry.hands.get(hand.handedness)
                    if h_state and h_state.current_pose != "UNKNOWN":
                        hold_tag = f" ({h_state.hold_duration_sec:.1f}s)" if h_state.is_holding else ""
                        text += f" | {h_state.current_pose}{hold_tag}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.44
                thickness = 1

                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                pad_x, pad_y = 6, 4

                box_x1 = badge_x
                box_y1 = badge_y - th - pad_y
                box_x2 = badge_x + tw + (pad_x * 2)
                box_y2 = badge_y + baseline + pad_y

                # Dark translucent badge background
                sub_img = canvas[max(0, box_y1):box_y2, max(0, box_x1):box_x2]
                if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
                    dark_rect = np.full_like(sub_img, 15)
                    cv2.addWeighted(dark_rect, 0.75, sub_img, 0.25, 0, sub_img)

                # Badge border and text
                border_col = (0, 255, 163) if is_mouse_hand else bone_color
                cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), border_col, thickness=1)
                cv2.putText(
                    canvas,
                    text,
                    (badge_x + pad_x, badge_y),
                    font,
                    font_scale,
                    self._bgr_white,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

        # ----------------------------------------------------------------------
        # 7. DYNAMIC & DUAL GESTURE BANNER NOTIFICATION (TOP CENTER HUD)
        # ----------------------------------------------------------------------
        if draw_hud and gesture_telemetry is not None:
            active_dyn = getattr(gesture_telemetry, "active_dynamic_gesture", None)
            active_dual = getattr(gesture_telemetry, "active_dual_gesture", None)

            banner_text = None
            banner_col = (0, 240, 255)  # Cyan

            if active_dyn:
                hand_tag = getattr(gesture_telemetry, "dynamic_gesture_hand", "")
                banner_text = f"◈ GESTURE ACTION: {active_dyn} [{hand_tag.upper()}] ◈"
                banner_col = (0, 255, 163) if "SWIPE" in active_dyn else (255, 128, 0)
            elif active_dual:
                banner_text = f"◈ DUAL INTERACTION: {active_dual} ◈"
                banner_col = (200, 0, 255)

            if banner_text:
                h_f, w_f = canvas.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                thick = 2
                (bw, bh), bl = cv2.getTextSize(banner_text, font, font_scale, thick)
                bx = (w_f - bw) // 2
                by = 45

                # Dark glowing background box
                sub = canvas[by - bh - 8:by + bl + 8, bx - 14:bx + bw + 14]
                if sub.shape[0] > 0 and sub.shape[1] > 0:
                    bg = np.full_like(sub, 12)
                    cv2.addWeighted(bg, 0.85, sub, 0.15, 0, sub)

                # Futuristic brackets & border
                cv2.rectangle(canvas, (bx - 14, by - bh - 8), (bx + bw + 14, by + bl + 8), banner_col, 2)
                cv2.putText(canvas, banner_text, (bx, by), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

        return canvas

    def close(self) -> None:
        """Release MediaPipe resources gracefully."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
                logger.info("MediaPipe HandLandmarker closed successfully.")
            except Exception as e:
                logger.warning("Exception during HandLandmarker close: %s", e)
            finally:
                self._landmarker = None
                self._is_initialized = False
                self._processor_right.reset()
                self._processor_left.reset()

