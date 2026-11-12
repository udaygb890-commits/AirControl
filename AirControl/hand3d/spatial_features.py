"""
AIRCONTROL — 3D SPATIAL FEATURES MODULE
Extracts scale- and rotation-invariant canonical hand features, 3D fingertip
distances, thumb-to-finger pinch metrics, finger spread, and 3D velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hand3d.skeleton import (
    FINGERTIP_IDS,
    JOINT_INDEX_MCP,
    JOINT_INDEX_TIP,
    JOINT_MIDDLE_MCP,
    JOINT_MIDDLE_TIP,
    JOINT_PINKY_MCP,
    JOINT_PINKY_TIP,
    JOINT_RING_TIP,
    JOINT_THUMB_TIP,
    JOINT_WRIST,
)


@dataclass
class SpatialFeatures:
    """Consolidated spatial metrics and canonical invariant representation."""
    palm_scale: float = 1.0                                      # Characteristic length (Wrist -> Mid MCP)
    canonical_joints: List[Tuple[float, float, float]] = field(default_factory=list)  # 21 normalized local coords

    # 3D Pinch Distances (Metric or normalized by palm scale)
    thumb_index_dist: float = 0.0
    thumb_middle_dist: float = 0.0
    thumb_ring_dist: float = 0.0
    thumb_pinky_dist: float = 0.0
    index_middle_dist: float = 0.0

    # Adjacent Fingertip Spreads
    spread_index_middle: float = 0.0
    spread_middle_ring: float = 0.0
    spread_ring_pinky: float = 0.0

    # 3D Velocity (m/s or units/s)
    velocity_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_3d: float = 0.0


class SpatialFeatureExtractor:
    """
    Computes Euclidean distances, canonical local coordinate projections,
    and 3D kinematic velocities from 3D hand joints.
    """

    def __init__(self):
        self._prev_center: Optional[Tuple[float, float, float]] = None
        self._prev_timestamp: Optional[float] = None

    @staticmethod
    def _dist3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """3D Euclidean distance with NaN protection."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        d_sq = dx * dx + dy * dy + dz * dz
        if math.isnan(d_sq) or math.isinf(d_sq) or d_sq < 0:
            return 0.0
        return math.sqrt(d_sq)

    @classmethod
    def calculate_canonical_frame(
        cls,
        joints_3d: List[Tuple[float, float, float]],
        palm_normal: Tuple[float, float, float],
    ) -> Tuple[float, List[Tuple[float, float, float]]]:
        """
        Transform 21 3D joints into an invariant canonical local coordinate frame:
        - Origin at Wrist (0, 0, 0)
        - Local Y axis points from Wrist to Middle MCP (scaled to length 1.0)
        - Local Z axis is the palm normal
        - Local X axis is Y_local x Z_local
        - All distances are divided by palm scale S = ||Middle MCP - Wrist||

        Returns:
            Tuple of (palm_scale, 21 canonical (x, y, z) tuples)
        """
        if len(joints_3d) < 21:
            return 1.0, [(0.0, 0.0, 0.0)] * 21

        wrist = joints_3d[JOINT_WRIST]
        mid_mcp = joints_3d[JOINT_MIDDLE_MCP]

        # Vector from wrist to middle MCP
        vy_raw = (mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1], mid_mcp[2] - wrist[2])
        scale_sq = vy_raw[0] * vy_raw[0] + vy_raw[1] * vy_raw[1] + vy_raw[2] * vy_raw[2]
        if math.isnan(scale_sq) or math.isinf(scale_sq) or scale_sq < 1e-12:
            scale = 1.0
        else:
            scale = math.sqrt(scale_sq)

        if scale < 1e-6 or math.isnan(scale) or math.isinf(scale):
            scale = 1.0

        # Unit Y
        inv_scale = 1.0 / scale
        y_axis = (vy_raw[0] * inv_scale, vy_raw[1] * inv_scale, vy_raw[2] * inv_scale)

        # Unit Z (Palm Normal)
        z_axis = palm_normal

        # Unit X = Y x Z
        x_axis = (
            y_axis[1] * z_axis[2] - y_axis[2] * z_axis[1],
            y_axis[2] * z_axis[0] - y_axis[0] * z_axis[2],
            y_axis[0] * z_axis[1] - y_axis[1] * z_axis[0],
        )
        x_sq = x_axis[0] * x_axis[0] + x_axis[1] * x_axis[1] + x_axis[2] * x_axis[2]
        if not (math.isnan(x_sq) or math.isinf(x_sq) or x_sq < 1e-12):
            inv_x = 1.0 / math.sqrt(x_sq)
            x_axis = (x_axis[0] * inv_x, x_axis[1] * inv_x, x_axis[2] * inv_x)
        else:
            x_axis = (1.0, 0.0, 0.0)

        # Project every joint into this canonical frame
        canonical_joints: List[Tuple[float, float, float]] = []
        for p in joints_3d:
            # Offset relative to wrist
            dx = p[0] - wrist[0]
            dy = p[1] - wrist[1]
            dz = p[2] - wrist[2]

            # Dot products with unit axes, normalized by palm scale
            cx = (dx * x_axis[0] + dy * x_axis[1] + dz * x_axis[2]) * inv_scale
            cy = (dx * y_axis[0] + dy * y_axis[1] + dz * y_axis[2]) * inv_scale
            cz = (dx * z_axis[0] + dy * z_axis[1] + dz * z_axis[2]) * inv_scale

            canonical_joints.append((cx, cy, cz))

        return scale, canonical_joints

    def extract(
        self,
        joints_3d: List[Tuple[float, float, float]],
        palm_normal: Tuple[float, float, float],
        timestamp: Optional[float] = None,
    ) -> SpatialFeatures:
        """
        Compute full spatial features, normalized distances, and 3D velocity.
        """
        if len(joints_3d) < 21:
            return SpatialFeatures()

        scale, canonical = self.calculate_canonical_frame(joints_3d, palm_normal)

        # Canonical fingertip distances (dimensionless, scale-invariant)
        p_thumb = canonical[JOINT_THUMB_TIP]
        p_index = canonical[JOINT_INDEX_TIP]
        p_middle = canonical[JOINT_MIDDLE_TIP]
        p_ring = canonical[JOINT_RING_TIP]
        p_pinky = canonical[JOINT_PINKY_TIP]

        t_idx_dist = self._dist3d(p_thumb, p_index)
        t_mid_dist = self._dist3d(p_thumb, p_middle)
        t_rng_dist = self._dist3d(p_thumb, p_ring)
        t_pky_dist = self._dist3d(p_thumb, p_pinky)
        idx_mid_dist = self._dist3d(p_index, p_middle)

        # Adjacent fingertip spread
        spread_idx_mid = idx_mid_dist
        spread_mid_rng = self._dist3d(p_middle, p_ring)
        spread_rng_pky = self._dist3d(p_ring, p_pinky)

        # 3D Palm Velocity
        palm_center = joints_3d[JOINT_MIDDLE_MCP]
        velocity = (0.0, 0.0, 0.0)
        speed = 0.0

        if self._prev_center is not None and timestamp is not None and self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 1e-4:
                vx = (palm_center[0] - self._prev_center[0]) / dt
                vy = (palm_center[1] - self._prev_center[1]) / dt
                vz = (palm_center[2] - self._prev_center[2]) / dt
                velocity = (vx, vy, vz)
                speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        self._prev_center = palm_center
        self._prev_timestamp = timestamp

        return SpatialFeatures(
            palm_scale=round(scale, 4),
            canonical_joints=canonical,
            thumb_index_dist=round(t_idx_dist, 3),
            thumb_middle_dist=round(t_mid_dist, 3),
            thumb_ring_dist=round(t_rng_dist, 3),
            thumb_pinky_dist=round(t_pky_dist, 3),
            index_middle_dist=round(idx_mid_dist, 3),
            spread_index_middle=round(spread_idx_mid, 3),
            spread_middle_ring=round(spread_mid_rng, 3),
            spread_ring_pinky=round(spread_rng_pky, 3),
            velocity_3d=(round(velocity[0], 3), round(velocity[1], 3), round(velocity[2], 3)),
            speed_3d=round(speed, 3),
        )

    def reset(self) -> None:
        """Reset temporal velocity state."""
        self._prev_center = None
        self._prev_timestamp = None

