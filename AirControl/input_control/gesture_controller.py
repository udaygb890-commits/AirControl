"""
AIRCONTROL — GESTURE CONTROLLER MODULE
Detects air-mouse interaction gestures: pinch left-click, pinch drag,
middle-finger right-click, two-finger vertical scroll, and clutch (fist freeze).
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import (
    CLUTCH_FIST_ENABLED,
    DRAG_HOLD_DELAY_SEC,
    INDEX_MIDDLE_CLICK_THRESHOLD,
    INDEX_MIDDLE_RELEASE_THRESHOLD,
    PINCH_CLICK_THRESHOLD,
    PINCH_RELEASE_THRESHOLD,
    RIGHT_PINCH_RELEASE_THRESHOLD,
    RIGHT_PINCH_THRESHOLD,
    SCROLL_DEADZONE,
    SCROLL_SENSITIVITY,
)
from vision.hand_tracker import HandData


@dataclass
class GestureEvent:
    """Actionable gesture state emitted per frame."""
    state: str                             # "HOVER", "CLICK_LEFT", "DRAGGING", "CLICK_RIGHT", "SCROLL", "CLUTCH"
    pinch_ratio: float = 1.0               # Scale-invariant thumb-index distance
    right_pinch_ratio: float = 1.0         # Scale-invariant thumb-middle distance
    scroll_delta: int = 0                  # Steps to scroll (+ up, - down)
    is_left_down: bool = False             # Mouse button 1 physically held
    is_clutched: bool = False              # Cursor position locked in space
    details: str = "Hover"


class GestureController:
    """State machine identifying discrete air-mouse gestures from 21 hand landmarks."""

    def __init__(self):
        self._is_pinching = False
        self._pinch_start_time: float = 0.0
        self._is_dragging = False

        self._is_right_pinching = False
        self._right_click_cooldown: float = 0.0

        self._prev_scroll_y: Optional[float] = None
        self._last_state: str = "HOVER"

    @staticmethod
    def _dist3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Euclidean 3D distance between two landmarks."""
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2
        )

    def _get_finger_extensions(self, lms: List[Tuple[float, float, float]]) -> Dict[str, bool]:
        """Determine whether each of the 5 fingers is extended or curled."""
        wrist = lms[0]

        def dist_wrist(idx):
            return self._dist3d(lms[idx], wrist)

        # Thumb: extended if tip (4) is further from pinky MCP (17) than thumb IP (3)
        thumb_tip_pinky = self._dist3d(lms[4], lms[17])
        thumb_ip_pinky = self._dist3d(lms[3], lms[17])
        thumb_ext = thumb_tip_pinky > thumb_ip_pinky * 1.15

        return {
            "Thumb": thumb_ext,
            "Index": dist_wrist(8) > dist_wrist(6),     # Tip 8 > PIP 6
            "Middle": dist_wrist(12) > dist_wrist(10),  # Tip 12 > PIP 10
            "Ring": dist_wrist(16) > dist_wrist(14),    # Tip 16 > PIP 14
            "Pinky": dist_wrist(20) > dist_wrist(18),   # Tip 20 > PIP 18
        }

    def process(self, hand: HandData) -> GestureEvent:
        """
        Evaluate hand geometry and return the active gesture command.

        Args:
            hand: Structured telemetry from HandTracker for the active mouse hand.

        Returns:
            GestureEvent detailing the requested mouse action.
        """
        lms = hand.landmarks_norm
        if len(lms) < 21:
            return GestureEvent(state="HOVER")

        now = time.perf_counter()

        # 1. Palm Scale (Distance between Wrist 0 and Middle MCP 9)
        # Used for distance-independent normalized ratio calculations
        palm_scale = max(0.01, self._dist3d(lms[0], lms[9]))

        # 2. Check Finger Extensions
        extensions = self._get_finger_extensions(lms)
        ext_count = sum(extensions.values())

        # ----------------------------------------------------------------------
        # GESTURE 1: CLUTCH (Fist / Freeze Cursor)
        # When all 4 main fingers (Index, Middle, Ring, Pinky) are curled into a fist,
        # cursor freeze engages like lifting a mouse to reposition.
        # ----------------------------------------------------------------------
        is_fist = (
            not extensions["Index"] and
            not extensions["Middle"] and
            not extensions["Ring"] and
            not extensions["Pinky"]
        )
        if CLUTCH_FIST_ENABLED and is_fist:
            self._reset_transient_states()
            return GestureEvent(
                state="CLUTCH",
                is_clutched=True,
                details="Clutch Engaged (Fist)",
            )

        # Calculate Normalized Distances
        # 1. Index finger to Middle finger distance (Tips & DIPs)
        idx_mid_tip = self._dist3d(lms[8], lms[12]) / palm_scale
        idx_mid_dip = self._dist3d(lms[7], lms[11]) / palm_scale
        idx_mid_touch_dist = min(idx_mid_tip, idx_mid_dip)

        # 2. Thumb to Index distance
        thumb_idx_dist = self._dist3d(lms[4], lms[8]) / palm_scale

        # 3. Thumb to Middle distance (Right Click)
        middle_pinch_dist = self._dist3d(lms[4], lms[12]) / palm_scale

        # ----------------------------------------------------------------------
        # GESTURE 2: LEFT CLICK & DRAG
        # Triggered when Middle finger touches Index finger (or Thumb-Index pinch)
        # ----------------------------------------------------------------------
        is_left_touch = (
            idx_mid_touch_dist < INDEX_MIDDLE_CLICK_THRESHOLD or
            thumb_idx_dist < PINCH_CLICK_THRESHOLD
        )
        is_left_release = (
            idx_mid_touch_dist > INDEX_MIDDLE_RELEASE_THRESHOLD and
            thumb_idx_dist > PINCH_RELEASE_THRESHOLD
        )

        effective_pinch_ratio = min(idx_mid_touch_dist, thumb_idx_dist)

        if is_left_touch:
            self._prev_scroll_y = None  # Cancel scroll if fingers touch
            if not self._is_pinching:
                # Touch contact started
                self._is_pinching = True
                self._pinch_start_time = now
                self._is_dragging = False

            hold_duration = now - self._pinch_start_time
            details_suffix = "Index-Middle Touch" if idx_mid_touch_dist < INDEX_MIDDLE_CLICK_THRESHOLD else "Pinch"

            if hold_duration >= DRAG_HOLD_DELAY_SEC:
                self._is_dragging = True
                return GestureEvent(
                    state="DRAGGING",
                    pinch_ratio=effective_pinch_ratio,
                    is_left_down=True,
                    details=f"Dragging ({details_suffix})",
                )
            else:
                return GestureEvent(
                    state="CLICK_LEFT",
                    pinch_ratio=effective_pinch_ratio,
                    is_left_down=False,
                    details=f"Left Click ({details_suffix})",
                )

        elif is_left_release:
            was_dragging = self._is_dragging
            was_pinching = self._is_pinching
            self._is_pinching = False
            self._is_dragging = False

            if was_pinching and not was_dragging:
                # Quick tap and release -> trigger click action
                return GestureEvent(
                    state="CLICK_LEFT",
                    pinch_ratio=effective_pinch_ratio,
                    is_left_down=False,
                    details="Single Left Click (Tap)",
                )

        # ----------------------------------------------------------------------
        # GESTURE 3: RIGHT CLICK (Thumb + Middle Finger Pinch)
        # ----------------------------------------------------------------------
        if middle_pinch_dist < RIGHT_PINCH_THRESHOLD:
            if not self._is_right_pinching and (now - self._right_click_cooldown > 0.40):
                self._is_right_pinching = True
                self._right_click_cooldown = now
                return GestureEvent(
                    state="CLICK_RIGHT",
                    right_pinch_ratio=middle_pinch_dist,
                    details="Right Click",
                )
        elif middle_pinch_dist > RIGHT_PINCH_RELEASE_THRESHOLD:
            self._is_right_pinching = False

        # ----------------------------------------------------------------------
        # GESTURE 4: TWO-FINGER SCROLL
        # Index & Middle extended side-by-side (with gap between them), Ring & Pinky curled.
        # ----------------------------------------------------------------------
        is_scroll_pose = (
            extensions["Index"] and
            extensions["Middle"] and
            not extensions["Ring"] and
            not extensions["Pinky"]
        )

        if is_scroll_pose and (INDEX_MIDDLE_RELEASE_THRESHOLD <= idx_mid_touch_dist < 0.45):
            curr_y = (lms[8][1] + lms[12][1]) / 2.0
            scroll_delta = 0

            if self._prev_scroll_y is not None:
                dy = curr_y - self._prev_scroll_y
                if abs(dy) > SCROLL_DEADZONE:
                    scroll_delta = int(-dy * SCROLL_SENSITIVITY * 100)

            self._prev_scroll_y = curr_y
            self._reset_transient_states()

            return GestureEvent(
                state="SCROLL",
                scroll_delta=scroll_delta,
                details=f"Scroll (Delta: {scroll_delta})",
            )
        else:
            self._prev_scroll_y = None

        # Default: normal hovering / cursor movement
        return GestureEvent(
            state="HOVER",
            pinch_ratio=effective_pinch_ratio,
            right_pinch_ratio=middle_pinch_dist,
            details="Hover",
        )

    def _reset_transient_states(self) -> None:
        """Reset pinch and click hold states when shifting to macro gestures."""
        self._is_pinching = False
        self._is_dragging = False
        self._is_right_pinching = False

    def reset(self) -> None:
        """Full reset of gesture memory on tracking drop."""
        self._reset_transient_states()
        self._prev_scroll_y = None
        self._right_click_cooldown = 0.0
