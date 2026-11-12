"""
AIRCONTROL — 3D TEMPORAL FILTER AND OUTLIER REJECTION MODULE
Applies adaptive One-Euro filtering and outlier rejection to 21 articulated
3D hand landmarks to eliminate jitter at rest while preserving zero lag during fast motion.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple


class LowPassFilter:
    """Standard first-order low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self) -> None:
        self.s = None


class OneEuroFilter1D:
    """
    1D One-Euro Filter (Casiez, Roussel, Vogel - CHI 2012).
    Dynamically adjusts cutoff frequency based on input rate of change.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filt.filter(x)

        dt = timestamp - self.last_time
        self.last_time = timestamp

        if dt <= 1e-5:
            dt = 1.0 / max(1.0, self.freq)

        # Estimate derivative of input signal
        prev_x = self.x_filt.s if self.x_filt.s is not None else x
        dx = (x - prev_x) / dt
        edx = self.dx_filt.filter(dx, self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt.filter(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None


class OneEuroFilter3D:
    """3D vector filter combining three 1D One-Euro filters."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ):
        self.fx = OneEuroFilter1D(freq, min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(freq, min_cutoff, beta, d_cutoff)
        self.fz = OneEuroFilter1D(freq, min_cutoff, beta, d_cutoff)

    def filter(
        self,
        pt: Tuple[float, float, float],
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        x = self.fx.filter(pt[0], timestamp)
        y = self.fy.filter(pt[1], timestamp)
        z = self.fz.filter(pt[2], timestamp)
        return (x, y, z)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


class HandFilter3D:
    """
    Manages 21 3D joint filters with outlier jump rejection and tracking-loss detection.
    """

    def __init__(
        self,
        max_jump_m: float = 0.15,  # Max allowable displacement per frame (in meters / normalized units)
        max_lost_frames: int = 5,
    ):
        self.max_jump = max_jump_m
        self.max_lost_frames = max_lost_frames
        self.lost_frames_count = 0

        self._filters: List[OneEuroFilter3D] = [
            OneEuroFilter3D() for _ in range(21)
        ]
        self._prev_joints: Optional[List[Tuple[float, float, float]]] = None

    def filter_joints(
        self,
        joints_3d: List[Tuple[float, float, float]],
        timestamp: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Filter 21 joints, rejecting unrealistic single-frame spikes and smoothing jitter.
        """
        if len(joints_3d) < 21:
            self.on_tracking_lost()
            return joints_3d

        self.lost_frames_count = 0
        filtered: List[Tuple[float, float, float]] = []

        for i in range(21):
            curr = joints_3d[i]

            # NaN / Inf protection
            if any(math.isnan(c) or math.isinf(c) for c in curr):
                if self._prev_joints is not None and len(self._prev_joints) > i:
                    curr = self._prev_joints[i]
                else:
                    curr = (0.0, 0.0, 0.0)

            # Outlier rejection check against previous position
            if self._prev_joints is not None and len(self._prev_joints) > i:
                prev = self._prev_joints[i]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                dz = curr[2] - prev[2]
                d_sq = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(d_sq) if not (math.isnan(d_sq) or d_sq < 0) else 0.0

                if dist > self.max_jump and dist > 1e-6:
                    # Clamp spike towards previous position
                    scale = self.max_jump / dist
                    curr = (
                        prev[0] + dx * scale,
                        prev[1] + dy * scale,
                        prev[2] + dz * scale,
                    )

            # Apply One-Euro filter
            f_pt = self._filters[i].filter(curr, timestamp)
            filtered.append(f_pt)

        self._prev_joints = list(filtered)
        return filtered

    def on_tracking_lost(self) -> None:
        """Call when hand is temporarily lost from camera frame."""
        self.lost_frames_count += 1
        if self.lost_frames_count >= self.max_lost_frames:
            self.reset()

    def reset(self) -> None:
        """Reset all joint filters and internal state."""
        for f in self._filters:
            f.reset()
        self._prev_joints = None
        self.lost_frames_count = 0

