"""
AIRCONTROL — AIR MOUSE MODULE
Coordinates relative hand tracking, exponential cursor smoothing,
gesture dispatching, and Windows OS mouse manipulation via PyAutoGUI.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import pyautogui

from config import (
    DEFAULT_AIR_MOUSE_ENABLED,
    DEFAULT_MOUSE_DEADZONE,
    DEFAULT_MOUSE_SENSITIVITY_X,
    DEFAULT_MOUSE_SENSITIVITY_Y,
    DEFAULT_SMOOTHING_FACTOR,
    PRIMARY_MOUSE_HAND,
)
from input_control.cursor_smoother import CursorSmoother
from input_control.gesture_controller import GestureController, GestureEvent
from vision.hand_tracker import HandData, HandTrackingResult

logger = logging.getLogger("AirControl.AirMouse")

# Optimize PyAutoGUI for real-time 60 FPS operation
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False


@dataclass
class AirMouseTelemetry:
    """Thread-safe snapshot of Air Mouse state for GUI rendering."""
    enabled: bool = False
    state: str = "IDLE"                    # "IDLE", "MOVING", "LEFT_CLICK", "RIGHT_CLICK", "DRAGGING", "SCROLL", "CLUTCH", "NO_HAND"
    hand_used: Optional[str] = None        # "Right" or "Left"
    raw_dx: float = 0.0                    # Normalized frame displacement
    raw_dy: float = 0.0
    smooth_vx: float = 0.0                 # Filtered pixel movement
    smooth_vy: float = 0.0
    cursor_x: int = 0                      # Current screen coordinates
    cursor_y: int = 0
    pinch_ratio: float = 1.0
    is_dragging: bool = False
    is_spatial_3d: bool = False
    depth_z: float = 0.0
    details: str = "Air Mouse Idle"


class AirMouse:
    """
    Main Air Mouse engine translating relative hand movement and gestures
    into responsive, jitter-free Windows cursor control.
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_AIR_MOUSE_ENABLED,
        primary_hand: str = PRIMARY_MOUSE_HAND,
        sensitivity_x: float = DEFAULT_MOUSE_SENSITIVITY_X,
        sensitivity_y: float = DEFAULT_MOUSE_SENSITIVITY_Y,
        deadzone: float = DEFAULT_MOUSE_DEADZONE,
        alpha: float = DEFAULT_SMOOTHING_FACTOR,
    ):
        self._enabled = enabled
        self._primary_hand = primary_hand

        self.smoother = CursorSmoother(
            sensitivity_x=sensitivity_x,
            sensitivity_y=sensitivity_y,
            deadzone=deadzone,
            alpha=alpha,
        )
        self.gesture_controller = GestureController()

        # Previous landmark coordinates (Index fingertip ID 8)
        self._prev_index_norm: Optional[Tuple[float, float]] = None
        self._prev_3d_pos: Optional[Tuple[float, float, float]] = None
        self._spatial_3d_enabled: bool = False
        self._spatial_sensitivity: float = 3500.0
        self._is_left_down: bool = False
        self._lock = threading.Lock()

        # Desktop screen dimensions
        try:
            self._screen_w, self._screen_h = pyautogui.size()
        except Exception:
            self._screen_w, self._screen_h = 1920, 1080

        # Latest telemetry container
        self._telemetry = AirMouseTelemetry(enabled=self._enabled)

    # --------------------------------------------------------------------------
    # PROPERTIES & CONFIGURATION
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
                self._release_mouse_buttons()
                self.reset()
            self._telemetry.enabled = val

    @property
    def primary_hand(self) -> str:
        with self._lock:
            return self._primary_hand

    @primary_hand.setter
    def primary_hand(self, val: str) -> None:
        with self._lock:
            self._primary_hand = val
            self.reset()

    @property
    def spatial_3d_enabled(self) -> bool:
        with self._lock:
            return self._spatial_3d_enabled

    @spatial_3d_enabled.setter
    def spatial_3d_enabled(self, val: bool) -> None:
        with self._lock:
            self._spatial_3d_enabled = bool(val)
            self._telemetry.is_spatial_3d = bool(val)
            self._prev_3d_pos = None

    @property
    def sensitivity(self) -> float:
        return self.smoother.sensitivity_x

    def set_sensitivity(self, val: float) -> None:
        with self._lock:
            self.smoother.sensitivity_x = val
            self.smoother.sensitivity_y = val

    @property
    def deadzone(self) -> float:
        return self.smoother.deadzone

    def set_deadzone(self, val: float) -> None:
        with self._lock:
            self.smoother.deadzone = val

    def get_telemetry(self) -> AirMouseTelemetry:
        """Retrieve thread-safe telemetry snapshot."""
        with self._lock:
            return self._telemetry

    # --------------------------------------------------------------------------
    # PROCESSING PIPELINE
    # --------------------------------------------------------------------------
    def process_frame(self, tracking_result: HandTrackingResult) -> AirMouseTelemetry:
        """
        Process detected hand landmarks and drive the Windows mouse cursor.

        Args:
            tracking_result: Output from HandTracker.process_frame().

        Returns:
            AirMouseTelemetry snapshot for GUI telemetry cards.
        """
        with self._lock:
            # 1. Select Active Mouse Hand
            active_hand = self._select_mouse_hand(tracking_result)

            if active_hand is None:
                self._handle_tracking_lost()
                return self._telemetry

            # 2. Extract Index Fingertip Landmark (MediaPipe ID 8)
            lms = active_hand.landmarks_norm
            if len(lms) < 21:
                self._handle_tracking_lost()
                return self._telemetry

            curr_idx_x, curr_idx_y, _ = lms[8]

            # 3. Calculate Relative Landmark Displacement
            raw_depth = 0.0
            if self._spatial_3d_enabled and active_hand.hand_model_3d and active_hand.hand_model_3d.is_valid:
                # 3D Metric Spatial Cursor Control
                curr_3d = active_hand.hand_model_3d.joints_filtered[8]
                raw_depth = curr_3d[2]
                if self._prev_3d_pos is None:
                    self._prev_3d_pos = curr_3d
                    self._prev_index_norm = (curr_idx_x, curr_idx_y)
                    self.smoother.reset()
                    self._telemetry.state = "ACQUIRED"
                    self._telemetry.hand_used = active_hand.handedness
                    self._telemetry.is_spatial_3d = True
                    self._telemetry.depth_z = round(raw_depth, 3)
                    self._telemetry.details = "3D Spatial tracking locked"
                    return self._telemetry

                # In world coordinates: X is right, Y is up, Z is depth
                # In screen coordinates: +X is right, +Y is down
                d_world_x = curr_3d[0] - self._prev_3d_pos[0]
                d_world_y = -(curr_3d[1] - self._prev_3d_pos[1])
                self._prev_3d_pos = curr_3d
                self._prev_index_norm = (curr_idx_x, curr_idx_y)

                raw_dx = (d_world_x * self._spatial_sensitivity) / self._screen_w
                raw_dy = (d_world_y * self._spatial_sensitivity) / self._screen_h
            else:
                if self._prev_index_norm is None:
                    # First frame after tracking acquired — seed position without moving cursor
                    self._prev_index_norm = (curr_idx_x, curr_idx_y)
                    self.smoother.reset()
                    self._telemetry.state = "ACQUIRED"
                    self._telemetry.hand_used = active_hand.handedness
                    self._telemetry.details = "Hand tracking locked"
                    return self._telemetry

                prev_x, prev_y = self._prev_index_norm
                raw_dx = curr_idx_x - prev_x
                raw_dy = curr_idx_y - prev_y
                self._prev_index_norm = (curr_idx_x, curr_idx_y)

            # 4. Gesture Recognition
            gesture_event: GestureEvent = self.gesture_controller.process(active_hand)

            # 5. Cursor Movement & Action Execution
            smooth_vx, smooth_vy = 0.0, 0.0
            state = "HOVER"
            details = gesture_event.details

            if gesture_event.state == "CLUTCH":
                # User made a fist: lock cursor position (safety repositioning)
                state = "CLUTCH"
                self.smoother.reset()
                self._release_mouse_buttons()

            elif gesture_event.state == "SCROLL":
                # Two-finger scroll
                state = "SCROLL"
                self.smoother.reset()
                if self._enabled and gesture_event.scroll_delta != 0:
                    try:
                        pyautogui.scroll(gesture_event.scroll_delta)
                    except Exception as e:
                        logger.debug("Scroll execution exception: %s", e)

            else:
                # Normal relative motion / click / drag
                smooth_vx, smooth_vy = self.smoother.smooth(
                    raw_dx=raw_dx,
                    raw_dy=raw_dy,
                    screen_w=self._screen_w,
                    screen_h=self._screen_h,
                )

                if abs(smooth_vx) > 0.01 or abs(smooth_vy) > 0.01:
                    state = "MOVING"

                # Handle Click Actions
                if gesture_event.state == "CLICK_LEFT":
                    state = "LEFT_CLICK"
                    if self._enabled and not self._is_left_down:
                        try:
                            pyautogui.click()
                        except Exception as e:
                            logger.debug("Click execution exception: %s", e)

                elif gesture_event.state == "DRAGGING":
                    state = "DRAGGING"
                    if self._enabled and not self._is_left_down:
                        try:
                            pyautogui.mouseDown(button="left")
                            self._is_left_down = True
                        except Exception as e:
                            logger.debug("MouseDown execution exception: %s", e)

                elif gesture_event.state == "CLICK_RIGHT":
                    state = "RIGHT_CLICK"
                    if self._enabled:
                        try:
                            pyautogui.rightClick()
                        except Exception as e:
                            logger.debug("Right click execution exception: %s", e)

                # Release left button if drag state dropped
                if not gesture_event.is_left_down and self._is_left_down:
                    self._release_mouse_buttons()

                # Execute Relative Movement
                if self._enabled and (abs(smooth_vx) > 0.0 or abs(smooth_vy) > 0.0):
                    self._execute_cursor_movement(smooth_vx, smooth_vy)

            # Query current screen cursor position
            try:
                cur_x, cur_y = pyautogui.position()
            except Exception:
                cur_x, cur_y = 0, 0

            # 6. Update Telemetry Record
            self._telemetry = AirMouseTelemetry(
                enabled=self._enabled,
                state=state,
                hand_used=active_hand.handedness,
                raw_dx=raw_dx,
                raw_dy=raw_dy,
                smooth_vx=smooth_vx,
                smooth_vy=smooth_vy,
                cursor_x=cur_x,
                cursor_y=cur_y,
                pinch_ratio=gesture_event.pinch_ratio,
                is_dragging=self._is_left_down,
                is_spatial_3d=self._spatial_3d_enabled,
                depth_z=round(raw_depth, 3),
                details=details,
            )
            return self._telemetry

    def _select_mouse_hand(self, tracking_result: HandTrackingResult) -> Optional[HandData]:
        """Choose the hand designated to control the mouse cursor."""
        if not tracking_result.hands:
            return None

        # Prefer user's designated primary hand
        if self._primary_hand != "Any":
            for hand in tracking_result.hands:
                if hand.handedness == self._primary_hand:
                    return hand

        # Fallback to first detected hand
        return tracking_result.hands[0]

    def _execute_cursor_movement(self, vx: float, vy: float) -> None:
        """Move cursor with boundary clamping."""
        try:
            curr_x, curr_y = pyautogui.position()
            target_x, target_y = self.smoother.clamp_to_screen(
                current_x=curr_x,
                current_y=curr_y,
                dx=vx,
                dy=vy,
                screen_w=self._screen_w,
                screen_h=self._screen_h,
            )
            pyautogui.moveTo(target_x, target_y)
        except Exception as e:
            logger.debug("Cursor moveTo execution exception: %s", e)

    def _release_mouse_buttons(self) -> None:
        """Ensure mouse buttons are released cleanly."""
        if self._is_left_down:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
            self._is_left_down = False

    def _handle_tracking_lost(self) -> None:
        """Handle tracking loss gracefully without cursor jumps."""
        self._release_mouse_buttons()
        self._prev_index_norm = None
        self._prev_3d_pos = None
        self.smoother.reset()
        self.gesture_controller.reset()

        self._telemetry.state = "NO_HAND"
        self._telemetry.raw_dx = 0.0
        self._telemetry.raw_dy = 0.0
        self._telemetry.smooth_vx = 0.0
        self._telemetry.smooth_vy = 0.0
        self._telemetry.details = "No hand detected in optical field"

    def reset(self) -> None:
        """Reset internal trackers on toggle or source change."""
        self._release_mouse_buttons()
        self._prev_index_norm = None
        self._prev_3d_pos = None
        self.smoother.reset()
        self.gesture_controller.reset()

    # Method alias for convenience
    update = process_frame

