"""
AIRCONTROL — 3D HAND CALIBRATION MODULE
Calibrates user-specific hand dimensions, baseline palm scale, and finger lengths
from a steady reference Open-Palm pose.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from hand3d.skeleton import (
    FINGER_JOINTS,
    JOINT_MIDDLE_MCP,
    JOINT_WRIST,
)

# Standard human hand defaults (approx. in meters for MediaPipe world landmarks)
DEFAULT_PALM_SCALE_M = 0.085  # ~8.5 cm wrist to middle MCP
DEFAULT_FINGER_LENGTHS_M = {
    "Thumb": 0.090,
    "Index": 0.095,
    "Middle": 0.105,
    "Ring": 0.098,
    "Pinky": 0.078,
}


@dataclass
class CalibrationProfile:
    """User-specific anatomical hand scale profile."""
    is_calibrated: bool = False
    palm_scale: float = DEFAULT_PALM_SCALE_M
    finger_lengths: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FINGER_LENGTHS_M))
    timestamp: float = 0.0
    sample_count: int = 0


class HandCalibrator:
    """
    Manages capturing, averaging, and applying personalized hand calibrations.
    """

    def __init__(self, target_samples: int = 30):
        self.target_samples = target_samples
        self.profile = CalibrationProfile()
        self.is_collecting: bool = False

        self._sample_palm_scales: List[float] = []
        self._sample_finger_lengths: Dict[str, List[float]] = {k: [] for k in DEFAULT_FINGER_LENGTHS_M}

    @staticmethod
    def _dist3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def start_calibration(self) -> None:
        """Begin collecting calibration frames."""
        self.is_collecting = True
        self._sample_palm_scales.clear()
        for k in self._sample_finger_lengths:
            self._sample_finger_lengths[k].clear()

    def add_sample(self, joints_3d: List[Tuple[float, float, float]]) -> bool:
        """
        Record a sample frame of an open palm.
        Returns True when target sample count is reached and calibration is completed.
        """
        if not self.is_collecting or len(joints_3d) < 21:
            return False

        # Measure palm scale (Wrist -> Middle MCP)
        wrist = joints_3d[JOINT_WRIST]
        mid_mcp = joints_3d[JOINT_MIDDLE_MCP]
        palm_scale = self._dist3d(wrist, mid_mcp)

        if palm_scale < 1e-4:
            return False

        self._sample_palm_scales.append(palm_scale)

        # Measure finger chain lengths
        for finger_name, joint_ids in FINGER_JOINTS.items():
            chain_len = 0.0
            for i in range(len(joint_ids) - 1):
                p1 = joints_3d[joint_ids[i]]
                p2 = joints_3d[joint_ids[i + 1]]
                chain_len += self._dist3d(p1, p2)
            self._sample_finger_lengths[finger_name].append(chain_len)

        if len(self._sample_palm_scales) >= self.target_samples:
            self.finalize_calibration()
            return True

        return False

    def finalize_calibration(self) -> CalibrationProfile:
        """Compute average calibration profile from collected samples."""
        if not self._sample_palm_scales:
            return self.profile

        avg_palm_scale = sum(self._sample_palm_scales) / len(self._sample_palm_scales)
        avg_finger_lengths = {}
        for k, samples in self._sample_finger_lengths.items():
            avg_finger_lengths[k] = sum(samples) / len(samples) if samples else DEFAULT_FINGER_LENGTHS_M[k]

        self.profile = CalibrationProfile(
            is_calibrated=True,
            palm_scale=round(avg_palm_scale, 4),
            finger_lengths={k: round(v, 4) for k, v in avg_finger_lengths.items()},
            timestamp=time.time(),
            sample_count=len(self._sample_palm_scales),
        )
        self.is_collecting = False
        return self.profile

    def get_scale_ratio(self, current_palm_scale: float) -> float:
        """
        Ratio of current palm scale relative to calibrated baseline.
        Useful for depth estimation / camera distance normalization.
        """
        baseline = self.profile.palm_scale if self.profile.is_calibrated else DEFAULT_PALM_SCALE_M
        if baseline < 1e-6:
            return 1.0
        return current_palm_scale / baseline

    def reset(self) -> None:
        """Reset calibration to factory defaults."""
        self.profile = CalibrationProfile()
        self.is_collecting = False
        self._sample_palm_scales.clear()
        for k in self._sample_finger_lengths:
            self._sample_finger_lengths[k].clear()

