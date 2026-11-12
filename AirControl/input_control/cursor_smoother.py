"""
AIRCONTROL — CURSOR SMOOTHER MODULE
Implements exponential moving average (EMA) smoothing, dead-zone filtering,
non-linear acceleration, speed clamping, and screen boundary limits.
"""

import math
from typing import Tuple

from config import (
    DEFAULT_MOUSE_DEADZONE,
    DEFAULT_MOUSE_SENSITIVITY_X,
    DEFAULT_MOUSE_SENSITIVITY_Y,
    DEFAULT_SMOOTHING_FACTOR,
    MOUSE_ACCELERATION_ENABLED,
    MOUSE_ACCELERATION_POWER,
    MOUSE_MAX_SPEED_PX,
)


class CursorSmoother:
    """
    Filters raw frame-to-frame landmark displacement to eliminate jitter,
    apply dead-zone thresholding, and produce silky-smooth relative mouse movement.
    """

    def __init__(
        self,
        sensitivity_x: float = DEFAULT_MOUSE_SENSITIVITY_X,
        sensitivity_y: float = DEFAULT_MOUSE_SENSITIVITY_Y,
        deadzone: float = DEFAULT_MOUSE_DEADZONE,
        alpha: float = DEFAULT_SMOOTHING_FACTOR,
        max_speed_px: float = MOUSE_MAX_SPEED_PX,
        acceleration_enabled: bool = MOUSE_ACCELERATION_ENABLED,
    ):
        self.sensitivity_x = sensitivity_x
        self.sensitivity_y = sensitivity_y
        self.deadzone = deadzone
        self.alpha = alpha
        self.max_speed_px = max_speed_px
        self.acceleration_enabled = acceleration_enabled

        self._prev_vx: float = 0.0
        self._prev_vy: float = 0.0

    def smooth(
        self,
        raw_dx: float,
        raw_dy: float,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[float, float]:
        """
        Smooth and scale normalized landmark displacement into pixel delta.

        Args:
            raw_dx: Frame-to-frame change in normalized X landmark coordinate.
            raw_dy: Frame-to-frame change in normalized Y landmark coordinate.
            screen_w: Width of the desktop display in pixels.
            screen_h: Height of the desktop display in pixels.

        Returns:
            (delta_x_px, delta_y_px) filtered pixel movements.
        """
        # 1. Deadzone Filtering
        distance = math.hypot(raw_dx, raw_dy)
        if distance < self.deadzone:
            self._prev_vx *= 0.5  # Soft decay to prevent sudden stop artifact
            self._prev_vy *= 0.5
            return 0.0, 0.0

        # Subtract deadzone smoothly from magnitude so movement doesn't jump at the boundary
        effective_dist = distance - self.deadzone
        ratio = effective_dist / distance
        norm_dx = raw_dx * ratio
        norm_dy = raw_dy * ratio

        # 2. Acceleration (Non-linear precision curve)
        # Small movements stay small for pinpoint targeting; fast flicks accelerate across screen
        accel = 1.0
        if self.acceleration_enabled:
            # Scale factor based on speed
            accel = 1.0 + math.pow(min(distance * 35.0, 4.0), MOUSE_ACCELERATION_POWER)

        # 3. Convert normalized displacement to screen pixels
        # Using base multiplier proportional to screen dimensions
        target_vx = norm_dx * screen_w * self.sensitivity_x * accel * 0.75
        target_vy = norm_dy * screen_h * self.sensitivity_y * accel * 0.75

        # 4. Exponential Moving Average (EMA) Smoothing
        # Dynamic alpha: slightly higher alpha on fast movements to reduce latency
        dynamic_alpha = min(1.0, self.alpha + min(0.35, distance * 8.0))
        smoothed_vx = (dynamic_alpha * target_vx) + ((1.0 - dynamic_alpha) * self._prev_vx)
        smoothed_vy = (dynamic_alpha * target_vy) + ((1.0 - dynamic_alpha) * self._prev_vy)

        # 5. Speed Clamping (Safety against tracking glitches)
        speed = math.hypot(smoothed_vx, smoothed_vy)
        if speed > self.max_speed_px and speed > 0:
            scale = self.max_speed_px / speed
            smoothed_vx *= scale
            smoothed_vy *= scale

        # Update historical velocity
        self._prev_vx = smoothed_vx
        self._prev_vy = smoothed_vy

        return smoothed_vx, smoothed_vy

    def clamp_to_screen(
        self,
        current_x: float,
        current_y: float,
        dx: float,
        dy: float,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[int, int]:
        """Apply displacement and clamp position strictly within display bounds."""
        new_x = max(0, min(screen_w - 1, int(round(current_x + dx))))
        new_y = max(0, min(screen_h - 1, int(round(current_y + dy))))
        return new_x, new_y

    def reset(self) -> None:
        """Reset smoothing history on tracking loss or mode toggle."""
        self._prev_vx = 0.0
        self._prev_vy = 0.0

