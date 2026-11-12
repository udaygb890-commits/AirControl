"""
AIRCONTROL — GESTURE ENGINE MODULE (STAGE 3)
Coordinates static pose recognition, temporal gesture analysis, dynamic motion
detection (swipes, wave, circle, push), dual-hand interactions, and debounced events.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from config import (
    CIRCLE_MIN_POINTS,
    CIRCLE_MIN_RADIUS,
    DEFAULT_GESTURE_ENGINE_ENABLED,
    DUAL_HAND_GESTURES_ENABLED,
    GESTURE_COOLDOWN_SEC,
    GESTURE_HOLD_THRESHOLD_SEC,
    GESTURE_STABILITY_FRAMES,
    PINCH_ZOOM_MIN_DELTA,
    SWIPE_MAX_TIME_SEC,
    SWIPE_MIN_DISTANCE,
    SWIPE_VELOCITY_THRESHOLD,
    WAVE_MIN_REVERSALS,
)
from gestures.gesture_classifier import (
    GestureClassification,
    GestureClassifier,
    StaticGesture,
)

if TYPE_CHECKING:
    from vision.hand_tracker import HandData, HandTrackingResult

logger = logging.getLogger("AirControl.GestureEngine")


class DynamicGesture(str, Enum):
    """Dynamic motion and action gestures recognized across time."""
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    SWIPE_UP = "SWIPE_UP"
    SWIPE_DOWN = "SWIPE_DOWN"
    WAVE = "WAVE"
    CIRCLE_CLOCKWISE = "CIRCLE_CLOCKWISE"
    CIRCLE_COUNTER_CLOCKWISE = "CIRCLE_COUNTER_CLOCKWISE"
    PUSH_TAP = "PUSH_TAP"
    NONE = "NONE"


class DualHandGesture(str, Enum):
    """Coordinated multi-hand interactions."""
    DUAL_OPEN_PALMS = "DUAL_OPEN_PALMS"
    DUAL_FISTS = "DUAL_FISTS"
    TWO_HANDS_X = "TWO_HANDS_X"
    PINCH_ZOOM_IN = "PINCH_ZOOM_IN"
    PINCH_ZOOM_OUT = "PINCH_ZOOM_OUT"
    FRAME_BOX = "FRAME_BOX"
    NONE = "NONE"


@dataclass
class HandHistoryPoint:
    """Historical sample for temporal gesture and trajectory analysis."""
    timestamp_ms: int
    time_sec: float
    palm_x: float                          # Normalized [0, 1]
    palm_y: float
    palm_z: float
    index_x: float
    index_y: float
    raw_pose: str
    confidence: float


@dataclass
class HandGestureState:
    """Real-time gesture state for an individual hand."""
    handedness: str                        # "Right" or "Left"
    current_pose: str                      # Member of StaticGesture
    confidence: float
    hold_duration_sec: float
    is_holding: bool
    pose_description: str
    finger_extensions: Dict[str, bool] = field(default_factory=dict)
    finger_curls: Dict[str, float] = field(default_factory=dict)
    palm_direction: str = "FACING_CAMERA"
    hand_orientation: str = "UP"
    hand_angle_deg: float = 0.0


@dataclass
class GestureEvent:
    """Dispatched gesture action event."""
    hand_label: str                        # "Right", "Left", or "Both"
    gesture_type: str                      # "STATIC", "DYNAMIC", "DUAL"
    gesture_name: str                      # e.g. "SWIPE_LEFT", "OPEN_PALM", "PINCH_ZOOM_OUT"
    confidence: float
    hold_duration: float
    details: str
    timestamp_ms: int = 0


@dataclass
class GestureTelemetry:
    """Thread-safe snapshot of all gesture states for UI and HUD rendering."""
    enabled: bool = True
    hands: Dict[str, HandGestureState] = field(default_factory=dict)
    active_dynamic_gesture: Optional[str] = None
    dynamic_gesture_details: Optional[str] = None
    dynamic_gesture_hand: Optional[str] = None
    active_dual_gesture: Optional[str] = None
    dual_gesture_details: Optional[str] = None
    recent_events: List[GestureEvent] = field(default_factory=list)


class GestureEngine:
    """
    Core Gesture Intelligence Engine.
    Processes HandTrackingResult frames to classify static hand poses,
    track temporal trajectories, detect dynamic motion gestures (swipes, circle, wave),
    and coordinate dual-hand interactions.
    """

    HISTORY_CAPACITY = 35  # ~1.1 seconds at 30 FPS

    def __init__(
        self,
        enabled: bool = DEFAULT_GESTURE_ENGINE_ENABLED,
        stability_frames: int = GESTURE_STABILITY_FRAMES,
        cooldown_sec: float = GESTURE_COOLDOWN_SEC,
    ):
        self._enabled = enabled
        self._stability_frames = stability_frames
        self._cooldown_sec = cooldown_sec

        self.classifier = GestureClassifier()
        self._lock = threading.RLock()

        # Temporal histories per hand: "Right" and "Left"
        self._histories: Dict[str, Deque[HandHistoryPoint]] = {
            "Right": collections.deque(maxlen=self.HISTORY_CAPACITY),
            "Left": collections.deque(maxlen=self.HISTORY_CAPACITY),
        }

        # Stability state per hand
        self._last_raw_poses: Dict[str, str] = {"Right": "", "Left": ""}
        self._raw_pose_counters: Dict[str, int] = {"Right": 0, "Left": 0}
        self._stable_poses: Dict[str, str] = {
            "Right": StaticGesture.UNKNOWN.value,
            "Left": StaticGesture.UNKNOWN.value,
        }
        self._pose_start_times: Dict[str, float] = {"Right": 0.0, "Left": 0.0}

        # Dynamic gesture cooldown management
        self._last_dynamic_trigger_time: float = 0.0
        self._last_dynamic_gesture: Optional[str] = None
        self._last_dynamic_gesture_hand: Optional[str] = None
        self._last_dynamic_gesture_details: Optional[str] = None
        self._dynamic_display_until: float = 0.0

        # Dual hand history
        self._dual_distance_history: Deque[Tuple[float, float]] = collections.deque(maxlen=20)
        self._last_dual_trigger_time: float = 0.0
        self._last_dual_gesture: Optional[str] = None
        self._last_dual_details: Optional[str] = None
        self._dual_display_until: float = 0.0

        # Event dispatching
        self._listeners: List[Callable[[GestureEvent], None]] = []
        self._recent_events: Deque[GestureEvent] = collections.deque(maxlen=15)

    # --------------------------------------------------------------------------
    # PROPERTIES
    # --------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        with self._lock:
            self._enabled = val
            if not val:
                self.reset()

    def add_listener(self, callback: Callable[[GestureEvent], None]) -> None:
        """Register an event listener for gesture activations."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[GestureEvent], None]) -> None:
        """Unregister an event listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def reset(self) -> None:
        """Clear temporal buffers and reset internal state."""
        with self._lock:
            for h in self._histories.values():
                h.clear()
            self._raw_pose_counters = {"Right": 0, "Left": 0}
            self._stable_poses = {"Right": StaticGesture.UNKNOWN.value, "Left": StaticGesture.UNKNOWN.value}
            self._pose_start_times = {"Right": 0.0, "Left": 0.0}
            self._dual_distance_history.clear()
            self._last_dynamic_gesture = None
            self._last_dual_gesture = None

    # --------------------------------------------------------------------------
    # PRIMARY PROCESSING PIPELINE
    # --------------------------------------------------------------------------
    def process(
        self,
        tracking_result: HandTrackingResult,
        timestamp_ms: Optional[int] = None,
    ) -> GestureTelemetry:
        """
        Evaluate tracking result to update gesture classifications,
        evaluate temporal trajectories, and detect motion / dual gestures.

        Args:
            tracking_result: Result from HandTracker.process_frame.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            GestureTelemetry snapshot ready for rendering and automation.
        """
        with self._lock:
            now_sec = time.perf_counter()
            if timestamp_ms is None:
                timestamp_ms = int(now_sec * 1000)

            if not self._enabled:
                return GestureTelemetry(enabled=False)

            hands_telemetry: Dict[str, HandGestureState] = {}
            detected_labels = set()

            # 1. Process each detected hand individually
            for hand in tracking_result.hands:
                h_label = hand.handedness
                detected_labels.add(h_label)

                # Classify static hand pose
                classification = self.classifier.classify(hand)

                # Add point to temporal trajectory history
                self._record_history(h_label, hand, classification, timestamp_ms, now_sec)

                # Apply temporal stability / debouncing
                stable_pose, hold_duration, is_holding = self._update_stability(
                    h_label, classification.name, now_sec, timestamp_ms
                )

                state = HandGestureState(
                    handedness=h_label,
                    current_pose=stable_pose,
                    confidence=classification.confidence,
                    hold_duration_sec=hold_duration,
                    is_holding=is_holding,
                    pose_description=classification.description,
                    finger_extensions=classification.finger_extensions,
                    finger_curls=classification.finger_curls,
                    palm_direction=classification.palm_direction,
                    hand_orientation=classification.hand_orientation,
                    hand_angle_deg=classification.hand_angle_deg,
                )
                hands_telemetry[h_label] = state

                # Evaluate single-hand dynamic motion gestures
                self._evaluate_dynamic_gestures(h_label, now_sec, timestamp_ms)

            # Clean up history for hands no longer in view
            for h_label in ("Right", "Left"):
                if h_label not in detected_labels:
                    if len(self._histories[h_label]) > 0:
                        self._histories[h_label].clear()
                        self._raw_pose_counters[h_label] = 0
                        self._stable_poses[h_label] = StaticGesture.UNKNOWN.value

            # 2. Process Dual-Hand Gestures
            if DUAL_HAND_GESTURES_ENABLED and len(tracking_result.hands) == 2:
                self._evaluate_dual_hand_gestures(
                    tracking_result.hands, hands_telemetry, now_sec, timestamp_ms
                )
            else:
                self._dual_distance_history.clear()

            # Check dynamic gesture display expiration
            active_dyn = self._last_dynamic_gesture if now_sec < self._dynamic_display_until else None
            active_dyn_details = self._last_dynamic_gesture_details if now_sec < self._dynamic_display_until else None
            active_dyn_hand = self._last_dynamic_gesture_hand if now_sec < self._dynamic_display_until else None

            active_dual = self._last_dual_gesture if now_sec < self._dual_display_until else None
            active_dual_details = self._last_dual_details if now_sec < self._dual_display_until else None

            return GestureTelemetry(
                enabled=True,
                hands=hands_telemetry,
                active_dynamic_gesture=active_dyn,
                dynamic_gesture_details=active_dyn_details,
                dynamic_gesture_hand=active_dyn_hand,
                active_dual_gesture=active_dual,
                dual_gesture_details=active_dual_details,
                recent_events=list(self._recent_events),
            )

    # --------------------------------------------------------------------------
    # TEMPORAL STABILITY & HOLD TRACKING
    # --------------------------------------------------------------------------
    def _record_history(
        self,
        hand_label: str,
        hand: HandData,
        classification: GestureClassification,
        timestamp_ms: int,
        time_sec: float,
    ) -> None:
        """Append sample to hand's circular trajectory history."""
        # Normalize palm center to [0, 1]
        wrist = hand.landmarks_norm[0]
        middle_mcp = hand.landmarks_norm[9]
        index_tip = hand.landmarks_norm[8]

        palm_norm_x = (wrist[0] + middle_mcp[0]) * 0.5
        palm_norm_y = (wrist[1] + middle_mcp[1]) * 0.5
        palm_norm_z = (wrist[2] + middle_mcp[2]) * 0.5

        point = HandHistoryPoint(
            timestamp_ms=timestamp_ms,
            time_sec=time_sec,
            palm_x=palm_norm_x,
            palm_y=palm_norm_y,
            palm_z=palm_norm_z,
            index_x=index_tip[0],
            index_y=index_tip[1],
            raw_pose=classification.name,
            confidence=classification.confidence,
        )
        self._histories[hand_label].append(point)

    def _update_stability(
        self,
        hand_label: str,
        raw_pose: str,
        now_sec: float,
        timestamp_ms: int,
    ) -> Tuple[str, float, bool]:
        """
        Filter static pose jitter using consecutive frame verification.
        Returns: (stable_pose_name, hold_duration_sec, is_holding_threshold_met)
        """
        last_raw = self._last_raw_poses[hand_label]
        if raw_pose == last_raw:
            self._raw_pose_counters[hand_label] += 1
        else:
            self._last_raw_poses[hand_label] = raw_pose
            self._raw_pose_counters[hand_label] = 1

        # Check if pose has stabilized over required frames
        if self._raw_pose_counters[hand_label] >= self._stability_frames:
            curr_stable = self._stable_poses[hand_label]
            if raw_pose != curr_stable:
                # Pose transition
                self._stable_poses[hand_label] = raw_pose
                self._pose_start_times[hand_label] = now_sec

                if raw_pose != StaticGesture.UNKNOWN.value:
                    self._dispatch_event(GestureEvent(
                        hand_label=hand_label,
                        gesture_type="STATIC",
                        gesture_name=raw_pose,
                        confidence=1.0,
                        hold_duration=0.0,
                        details=f"Pose locked: {raw_pose}",
                        timestamp_ms=timestamp_ms,
                    ))

        stable_pose = self._stable_poses[hand_label]
        hold_time = max(0.0, now_sec - self._pose_start_times[hand_label])
        is_holding = hold_time >= GESTURE_HOLD_THRESHOLD_SEC and stable_pose != StaticGesture.UNKNOWN.value

        return stable_pose, hold_time, is_holding

    # --------------------------------------------------------------------------
    # DYNAMIC GESTURE EVALUATION (SWIPES, WAVE, CIRCLE, PUSH)
    # --------------------------------------------------------------------------
    def _evaluate_dynamic_gestures(
        self,
        hand_label: str,
        now_sec: float,
        timestamp_ms: int,
    ) -> None:
        """Inspect trajectory buffer for swipes, circular motions, wave, and push."""
        history = self._histories[hand_label]
        if len(history) < 6:
            return

        # Cooldown guard
        if now_sec - self._last_dynamic_trigger_time < self._cooldown_sec:
            return

        # 1. Swipes (Left, Right, Up, Down)
        swipe = self._detect_swipe(history, now_sec)
        if swipe != DynamicGesture.NONE:
            self._trigger_dynamic_gesture(
                swipe.value,
                hand_label,
                f"Dynamic {swipe.value} detected on {hand_label} Hand",
                now_sec,
                timestamp_ms,
            )
            return

        # 2. Wave Gesture
        if self._detect_wave(history):
            self._trigger_dynamic_gesture(
                DynamicGesture.WAVE.value,
                hand_label,
                f"Hand Wave detected on {hand_label} Hand",
                now_sec,
                timestamp_ms,
            )
            return

        # 3. Circular Motion (Clockwise / Counter-Clockwise)
        circle = self._detect_circle(history)
        if circle != DynamicGesture.NONE:
            self._trigger_dynamic_gesture(
                circle.value,
                hand_label,
                f"{circle.value} detected on {hand_label} Hand",
                now_sec,
                timestamp_ms,
            )
            return

        # 4. Push Tap (Forward z-movement)
        if self._detect_push(history):
            self._trigger_dynamic_gesture(
                DynamicGesture.PUSH_TAP.value,
                hand_label,
                f"Forward Push-Tap detected on {hand_label} Hand",
                now_sec,
                timestamp_ms,
            )
            return

    def _detect_swipe(self, history: Deque[HandHistoryPoint], now_sec: float) -> DynamicGesture:
        """Detect rapid horizontal or vertical swipe trajectory."""
        newest = history[-1]
        target_pts = [p for p in history if (now_sec - p.time_sec) <= SWIPE_MAX_TIME_SEC]
        if len(target_pts) < 4:
            return DynamicGesture.NONE

        oldest = target_pts[0]
        dt = max(0.01, newest.time_sec - oldest.time_sec)
        dx = newest.palm_x - oldest.palm_x
        dy = newest.palm_y - oldest.palm_y

        dist_x = abs(dx)
        dist_y = abs(dy)
        vel_x = dist_x / dt
        vel_y = dist_y / dt

        # Horizontal Swipe: dominant x-axis travel
        if dist_x >= SWIPE_MIN_DISTANCE and vel_x >= SWIPE_VELOCITY_THRESHOLD and dist_x > 1.7 * dist_y:
            return DynamicGesture.SWIPE_RIGHT if dx > 0 else DynamicGesture.SWIPE_LEFT

        # Vertical Swipe: dominant y-axis travel
        if dist_y >= SWIPE_MIN_DISTANCE and vel_y >= SWIPE_VELOCITY_THRESHOLD and dist_y > 1.7 * dist_x:
            # Note: y points DOWN in image coordinates, so dy < 0 is upward
            return DynamicGesture.SWIPE_DOWN if dy > 0 else DynamicGesture.SWIPE_UP

        return DynamicGesture.NONE

    def _detect_wave(self, history: Deque[HandHistoryPoint]) -> bool:
        """Detect rhythmic left-right palm waving."""
        if len(history) < 14:
            return False

        # Waving is predominantly performed with an OPEN_PALM
        recent_open = sum(1 for p in list(history)[-10:] if p.raw_pose == StaticGesture.OPEN_PALM.value)
        if recent_open < 6:
            return False

        # Count direction reversals along x-axis
        xs = [p.palm_x for p in history]
        diffs = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        significant_diffs = [d for d in diffs if abs(d) > 0.008]

        if len(significant_diffs) < 5:
            return False

        reversals = 0
        for i in range(1, len(significant_diffs)):
            if (significant_diffs[i] > 0 and significant_diffs[i - 1] < 0) or (
                significant_diffs[i] < 0 and significant_diffs[i - 1] > 0
            ):
                reversals += 1

        return reversals >= WAVE_MIN_REVERSALS

    def _detect_circle(self, history: Deque[HandHistoryPoint]) -> DynamicGesture:
        """Detect circular trajectory using angular accumulation around path centroid."""
        if len(history) < CIRCLE_MIN_POINTS:
            return DynamicGesture.NONE

        pts = [(p.index_x, p.index_y) for p in history]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Centroid
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        # Average radius
        radii = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in pts]
        avg_r = sum(radii) / len(radii)
        if avg_r < CIRCLE_MIN_RADIUS:
            return DynamicGesture.NONE

        # Calculate cumulative angular sweep
        angles = [math.atan2(y - cy, x - cx) for x, y in pts]
        total_angle = 0.0

        for i in range(1, len(angles)):
            d_theta = angles[i] - angles[i - 1]
            # Normalize angle wrap-around
            while d_theta > math.pi:
                d_theta -= 2 * math.pi
            while d_theta < -math.pi:
                d_theta += 2 * math.pi
            total_angle += d_theta

        # 270 degrees sweep in radians ~ 4.71 rad
        threshold_rad = 4.5
        if total_angle >= threshold_rad:
            # Positive angular sweep in standard math coordinates corresponds to CW in screen coords
            return DynamicGesture.CIRCLE_CLOCKWISE
        elif total_angle <= -threshold_rad:
            return DynamicGesture.CIRCLE_COUNTER_CLOCKWISE

        return DynamicGesture.NONE

    def _detect_push(self, history: Deque[HandHistoryPoint]) -> bool:
        """Detect rapid forward displacement along z-axis towards camera."""
        if len(history) < 6:
            return False

        oldest = history[-6]
        newest = history[-1]

        # MediaPipe z decreases as hand approaches camera
        dz = newest.palm_z - oldest.palm_z
        dt = max(0.01, newest.time_sec - oldest.time_sec)
        z_vel = -dz / dt  # Positive when moving toward camera

        return dz < -0.075 and z_vel > 0.25

    def _trigger_dynamic_gesture(
        self,
        gesture_name: str,
        hand_label: str,
        details: str,
        now_sec: float,
        timestamp_ms: int,
    ) -> None:
        """Record dynamic gesture activation and dispatch event."""
        self._last_dynamic_trigger_time = now_sec
        self._last_dynamic_gesture = gesture_name
        self._last_dynamic_gesture_hand = hand_label
        self._last_dynamic_gesture_details = details
        self._dynamic_display_until = now_sec + 1.25  # Display for 1.25s

        self._dispatch_event(GestureEvent(
            hand_label=hand_label,
            gesture_type="DYNAMIC",
            gesture_name=gesture_name,
            confidence=0.92,
            hold_duration=0.0,
            details=details,
            timestamp_ms=timestamp_ms,
        ))

    # --------------------------------------------------------------------------
    # DUAL-HAND INTERACTION EVALUATION
    # --------------------------------------------------------------------------
    def _evaluate_dual_hand_gestures(
        self,
        hands: List[HandData],
        hands_telemetry: Dict[str, HandGestureState],
        now_sec: float,
        timestamp_ms: int,
    ) -> None:
        """Detect coordinated two-hand poses, frame gesture, and pinch-zoom."""
        h1, h2 = hands[0], hands[1]
        p1 = h1.palm_center
        p2 = h2.palm_center

        # Normalized distance between palms
        dx = (p1[0] - p2[0]) / 1280.0
        dy = (p1[1] - p2[1]) / 720.0
        dist = math.sqrt(dx ** 2 + dy ** 2)

        self._dual_distance_history.append((now_sec, dist))

        # Cooldown guard for dual events
        if now_sec - self._last_dual_trigger_time < self._cooldown_sec:
            return

        pose1 = hands_telemetry.get(h1.handedness, None)
        pose2 = hands_telemetry.get(h2.handedness, None)

        if not pose1 or not pose2:
            return

        name1 = pose1.current_pose
        name2 = pose2.current_pose

        # 1. Dual Open Palms
        if name1 == StaticGesture.OPEN_PALM.value and name2 == StaticGesture.OPEN_PALM.value:
            self._trigger_dual_gesture(
                DualHandGesture.DUAL_OPEN_PALMS.value,
                "Both hands showing Open Palms",
                now_sec,
                timestamp_ms,
            )
            return

        # 2. Dual Fists
        if name1 == StaticGesture.FIST.value and name2 == StaticGesture.FIST.value:
            self._trigger_dual_gesture(
                DualHandGesture.DUAL_FISTS.value,
                "Both hands clenched in Fists",
                now_sec,
                timestamp_ms,
            )
            return

        # 3. Two Hands 'X' Formation (Crossing arms/wrists to Close Window)
        if len(h1.landmarks_norm) >= 21 and len(h2.landmarks_norm) >= 21:
            w1, t1 = h1.landmarks_norm[0], h1.landmarks_norm[12]
            w2, t2 = h2.landmarks_norm[0], h2.landmarks_norm[12]

            def _ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            def _intersect(A, B, C, D):
                return _ccw(A, B, C) != _ccw(A, B, D) and _ccw(A, C, D) != _ccw(B, C, D)

            is_crossed = _intersect(w1, t1, w2, t2)
            wrist_dist = math.sqrt((w1[0] - w2[0]) ** 2 + (w1[1] - w2[1]) ** 2)

            if is_crossed or (wrist_dist < 0.16 and abs(w1[0] - w2[0]) < 0.14):
                v1 = (t1[0] - w1[0], t1[1] - w1[1])
                v2 = (t2[0] - w2[0], t2[1] - w2[1])
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                if m1 > 1e-4 and m2 > 1e-4:
                    cos_a = np.clip(dot / (m1 * m2), -1.0, 1.0)
                    angle_deg = math.degrees(math.acos(cos_a))
                    if 20.0 <= angle_deg <= 165.0:
                        self._trigger_dual_gesture(
                            DualHandGesture.TWO_HANDS_X.value,
                            f"Two Hands Crossed in 'X' Formation ({angle_deg:.0f}°)",
                            now_sec,
                            timestamp_ms,
                        )
                        return

        # 3. Two-Hand Dynamic Zoom (Expanding or Contracting Palms)
        if len(self._dual_distance_history) >= 8:
            old_time, old_dist = self._dual_distance_history[0]
            dt = max(0.01, now_sec - old_time)
            d_dist = dist - old_dist

            if dt < 0.60:
                if d_dist > PINCH_ZOOM_MIN_DELTA:
                    self._trigger_dual_gesture(
                        DualHandGesture.PINCH_ZOOM_OUT.value,
                        f"Hands expanding outward (Zoom In +{d_dist:.2f})",
                        now_sec,
                        timestamp_ms,
                    )
                    self._dual_distance_history.clear()
                    return
                elif d_dist < -PINCH_ZOOM_MIN_DELTA:
                    self._trigger_dual_gesture(
                        DualHandGesture.PINCH_ZOOM_IN.value,
                        f"Hands contracting inward (Zoom Out {d_dist:.2f})",
                        now_sec,
                        timestamp_ms,
                    )
                    self._dual_distance_history.clear()
                    return

    def _trigger_dual_gesture(
        self,
        gesture_name: str,
        details: str,
        now_sec: float,
        timestamp_ms: int,
    ) -> None:
        """Record dual-hand gesture activation."""
        self._last_dual_trigger_time = now_sec
        self._last_dual_gesture = gesture_name
        self._last_dual_details = details
        self._dual_display_until = now_sec + 1.25

        self._dispatch_event(GestureEvent(
            hand_label="Both",
            gesture_type="DUAL",
            gesture_name=gesture_name,
            confidence=0.95,
            hold_duration=0.0,
            details=details,
            timestamp_ms=timestamp_ms,
        ))

    # --------------------------------------------------------------------------
    # EVENT DISPATCH
    # --------------------------------------------------------------------------
    def _dispatch_event(self, event: GestureEvent) -> None:
        """Append to recent event log and notify registered listeners."""
        self._recent_events.append(event)
        logger.info("Gesture Event: [%s] %s (%s)", event.gesture_type, event.gesture_name, event.details)

        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as e:
                logger.debug("Error in gesture listener callback: %s", e)
