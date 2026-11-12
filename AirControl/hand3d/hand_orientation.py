"""
AIRCONTROL — 3D HAND ORIENTATION MODULE
Calculates 3D palm normal vector, Euler angles (pitch, yaw, roll),
discrete facing directions, and temporal rotational velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from hand3d.skeleton import (
    JOINT_INDEX_MCP,
    JOINT_MIDDLE_MCP,
    JOINT_PINKY_MCP,
    JOINT_WRIST,
)


class PalmFacing(str, Enum):
    """Discrete cardinal facing direction of the palm surface."""
    FACING_CAMERA = "FACING_CAMERA"
    FACING_AWAY = "FACING_AWAY"
    FACING_UP = "FACING_UP"
    FACING_DOWN = "FACING_DOWN"
    FACING_LEFT = "FACING_LEFT"
    FACING_RIGHT = "FACING_RIGHT"
    UNKNOWN = "UNKNOWN"


@dataclass
class HandOrientation:
    """Consolidated 3D spatial orientation state of a hand."""
    palm_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # Unit normal vector
    palm_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Approximate center of palm
    pitch_deg: float = 0.0                                     # Pitch angle in degrees [-90, 90]
    yaw_deg: float = 0.0                                       # Yaw angle in degrees [-180, 180]
    roll_deg: float = 0.0                                      # Roll angle in degrees [-180, 180]
    facing: PalmFacing = PalmFacing.FACING_CAMERA
    angular_speed_deg_per_sec: float = 0.0                     # Rotational speed magnitude


class OrientationEstimator:
    """
    Computes 3D palm normal, coordinate frame, and Euler angles
    from 3D skeletal landmarks.
    """

    def __init__(self, is_right_hand: bool = True):
        self.is_right_hand = is_right_hand
        self._prev_normal: Optional[Tuple[float, float, float]] = None
        self._prev_timestamp: Optional[float] = None

    @staticmethod
    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Compute cross product of 3D vectors a x b."""
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Return unit vector. Falls back to (0, 0, 1) if length is 0 or NaN."""
        mag_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if math.isnan(mag_sq) or math.isinf(mag_sq) or mag_sq < 1e-12:
            return (0.0, 0.0, 1.0)
        mag = math.sqrt(mag_sq)
        inv = 1.0 / mag
        return (v[0] * inv, v[1] * inv, v[2] * inv)

    @classmethod
    def calculate_palm_normal(
        cls,
        joints_3d: List[Tuple[float, float, float]],
        is_right_hand: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Calculate unit normal vector emerging perpendicular from the palm face.

        Uses longitudinal vector (Wrist -> Middle MCP) and transverse vector
        (Index MCP -> Pinky MCP) to define the palm plane.
        """
        if len(joints_3d) < 21:
            return (0.0, 0.0, 1.0)

        wrist = joints_3d[JOINT_WRIST]
        mid_mcp = joints_3d[JOINT_MIDDLE_MCP]
        idx_mcp = joints_3d[JOINT_INDEX_MCP]
        pky_mcp = joints_3d[JOINT_PINKY_MCP]

        # Longitudinal vector: wrist -> middle MCP (points toward fingers)
        v_long = (mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1], mid_mcp[2] - wrist[2])

        # Transverse vector: index MCP -> pinky MCP
        v_trans = (pky_mcp[0] - idx_mcp[0], pky_mcp[1] - idx_mcp[1], pky_mcp[2] - idx_mcp[2])

        if is_right_hand:
            # v_trans x v_long points outward from right palm
            raw_normal = cls._cross(v_trans, v_long)
        else:
            # v_long x v_trans points outward from left palm
            raw_normal = cls._cross(v_long, v_trans)

        return cls._normalize(raw_normal)

    @classmethod
    def calculate_palm_center(
        cls,
        joints_3d: List[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        """Calculate geometric center of palm (Wrist + 4 MCPs centroid)."""
        if len(joints_3d) < 21:
            return (0.0, 0.0, 0.0)

        pts = [
            joints_3d[JOINT_WRIST],
            joints_3d[JOINT_INDEX_MCP],
            joints_3d[JOINT_MIDDLE_MCP],
            joints_3d[JOINT_PINKY_MCP],
        ]
        count = len(pts)
        cx = sum(p[0] for p in pts) / count
        cy = sum(p[1] for p in pts) / count
        cz = sum(p[2] for p in pts) / count
        return (cx, cy, cz)

    @classmethod
    def calculate_euler_angles(
        cls,
        joints_3d: List[Tuple[float, float, float]],
        palm_normal: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Derive pitch, yaw, roll Euler angles in degrees from the hand orientation.
        """
        if len(joints_3d) < 21:
            return 0.0, 0.0, 0.0

        wrist = joints_3d[JOINT_WRIST]
        mid_mcp = joints_3d[JOINT_MIDDLE_MCP]

        # Longitudinal vector Y_local (Wrist -> Middle MCP)
        vy = (mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1], mid_mcp[2] - wrist[2])
        vy = cls._normalize(vy)

        # Normal vector Z_local
        vz = palm_normal

        # Pitch: elevation of longitudinal axis relative to horizontal plane
        vy1_val = max(-1.0, min(1.0, -vy[1])) if not (math.isnan(vy[1]) or math.isinf(vy[1])) else 0.0
        pitch = math.degrees(math.asin(vy1_val))

        # Yaw: direction normal is pointing in XZ plane
        yaw = math.degrees(math.atan2(vz[0], vz[2])) if not (math.isnan(vz[0]) or math.isnan(vz[2])) else 0.0

        # Roll: rotation around longitudinal axis
        # Compute transverse vector X_local = Y_local x Z_local
        vx = cls._cross(vy, vz)
        roll = math.degrees(math.atan2(vx[1], vz[1])) if not (math.isnan(vx[1]) or math.isnan(vz[1])) else 0.0

        return round(pitch, 1), round(yaw, 1), round(roll, 1)

    @staticmethod
    def classify_facing_direction(normal: Tuple[float, float, float]) -> PalmFacing:
        """
        Classify the primary facing direction of the palm surface based on the
        dominant component of the normal vector.
        In camera coordinates: +Z is towards camera, -Z is into screen,
        +Y is down, -Y is up (or vice-versa in standard vision/screen convention).
        """
        nx, ny, nz = normal
        abs_x, abs_y, abs_z = abs(nx), abs(ny), abs(nz)

        # Dominant axis
        if abs_z >= abs_x and abs_z >= abs_y:
            # Z dominant
            # If nz > 0, normal faces toward the camera/viewer
            return PalmFacing.FACING_CAMERA if nz > 0 else PalmFacing.FACING_AWAY
        elif abs_y >= abs_x and abs_y >= abs_z:
            # Y dominant
            # In image/screen coords, negative Y is upward
            return PalmFacing.FACING_UP if ny < 0 else PalmFacing.FACING_DOWN
        else:
            # X dominant
            return PalmFacing.FACING_RIGHT if nx > 0 else PalmFacing.FACING_LEFT

    def estimate(
        self,
        joints_3d: List[Tuple[float, float, float]],
        timestamp: Optional[float] = None,
    ) -> HandOrientation:
        """
        Compute full 3D orientation state including angular speed tracking.
        """
        if len(joints_3d) < 21:
            return HandOrientation()

        normal = self.calculate_palm_normal(joints_3d, self.is_right_hand)
        center = self.calculate_palm_center(joints_3d)
        pitch, yaw, roll = self.calculate_euler_angles(joints_3d, normal)
        facing = self.classify_facing_direction(normal)

        # Angular velocity calculation
        angular_speed = 0.0
        if self._prev_normal is not None and timestamp is not None and self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 1e-4:
                # Dot product between consecutive normals
                dot = sum(a * b for a, b in zip(normal, self._prev_normal))
                dot_clamped = max(-1.0, min(1.0, dot))
                d_angle = math.degrees(math.acos(dot_clamped))
                angular_speed = d_angle / dt

        self._prev_normal = normal
        self._prev_timestamp = timestamp

        return HandOrientation(
            palm_normal=normal,
            palm_center=center,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            facing=facing,
            angular_speed_deg_per_sec=round(angular_speed, 1),
        )

    def reset(self) -> None:
        """Reset temporal state."""
        self._prev_normal = None
        self._prev_timestamp = None

