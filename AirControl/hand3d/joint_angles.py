"""
AIRCONTROL — 3D JOINT ANGLES MODULE
Calculates anatomical joint flexion/extension, finger abduction (spread),
and thumb opposition angles in 3D Euclidean space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from hand3d.skeleton import (
    FINGER_JOINTS,
    JOINT_INDEX_MCP,
    JOINT_MIDDLE_MCP,
    JOINT_PINKY_MCP,
    JOINT_RING_MCP,
    JOINT_THUMB_CMC,
    JOINT_THUMB_IP,
    JOINT_THUMB_MCP,
    JOINT_THUMB_TIP,
    JOINT_WRIST,
)


@dataclass
class FingerAngles:
    """Joint angles for an individual articulated digit."""
    finger_name: str
    mcp_flexion_deg: float = 0.0   # Flexion at Metacarpophalangeal joint
    pip_flexion_deg: float = 0.0   # Flexion at Proximal Interphalangeal joint
    dip_flexion_deg: float = 0.0   # Flexion at Distal Interphalangeal joint
    total_flexion_deg: float = 0.0 # Sum of flexions (0 = fully straight, >160 = curled)
    abduction_deg: float = 0.0     # Spread angle relative to adjacent reference finger
    is_extended: bool = True       # True if digit is extended straight


@dataclass
class HandJointAngles:
    """Consolidated 3D joint angle profile for an entire hand."""
    fingers: Dict[str, FingerAngles] = field(default_factory=dict)
    thumb_opposition_deg: float = 0.0  # Angle of thumb opposing across palm plane
    thumb_abduction_deg: float = 0.0   # Spread angle between Thumb and Index MCP
    index_middle_spread_deg: float = 0.0
    middle_ring_spread_deg: float = 0.0
    ring_pinky_spread_deg: float = 0.0


class JointAngleCalculator:
    """
    Computes 3D joint articulation angles from 21 joint landmark coordinates.
    Angles are expressed in degrees, where 0° denotes an anatomically straight joint.
    """

    @staticmethod
    def _vector(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Vector from p1 to p2."""
        return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

    @staticmethod
    def _dot(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

    @staticmethod
    def _magnitude(v: Tuple[float, float, float]) -> float:
        v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if math.isnan(v_sq) or v_sq < 0:
            return 0.0
        return math.sqrt(v_sq)

    @classmethod
    def angle_between_vectors(
        cls,
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
    ) -> float:
        """
        Compute angle between two 3D vectors in degrees [0.0, 180.0].
        """
        mag1 = cls._magnitude(v1)
        mag2 = cls._magnitude(v2)
        if mag1 < 1e-6 or mag2 < 1e-6 or math.isnan(mag1) or math.isnan(mag2):
            return 0.0

        dot = cls._dot(v1, v2)
        if math.isnan(dot):
            return 0.0
        cos_val = dot / (mag1 * mag2)
        cos_clamped = max(-1.0, min(1.0, cos_val))
        return math.degrees(math.acos(cos_clamped))

    @classmethod
    def joint_flexion_angle(
        cls,
        p_prev: Tuple[float, float, float],
        p_joint: Tuple[float, float, float],
        p_next: Tuple[float, float, float],
    ) -> float:
        """
        Compute flexion angle at p_joint formed by incoming bone (p_prev -> p_joint)
        and outgoing bone (p_joint -> p_next).
        Returns 0° when the two bones are collinearly aligned (straight).
        """
        v_in = cls._vector(p_prev, p_joint)
        v_out = cls._vector(p_joint, p_next)
        # Angle between incoming bone direction and outgoing bone direction
        # Collinear (straight) = 0°, 90° bend = 90° flexion
        return cls.angle_between_vectors(v_in, v_out)

    @classmethod
    def calculate_hand_angles(
        cls,
        joints_3d: List[Tuple[float, float, float]],
    ) -> HandJointAngles:
        """
        Calculate complete joint articulation angles for all 5 digits.

        Args:
            joints_3d: List of 21 3D joint coordinates (x, y, z).

        Returns:
            HandJointAngles structure.
        """
        if len(joints_3d) < 21:
            return HandJointAngles()

        wrist = joints_3d[JOINT_WRIST]
        angles_map: Dict[str, FingerAngles] = {}

        # 1. Four Main Fingers (Index, Middle, Ring, Pinky)
        for finger_name in ("Index", "Middle", "Ring", "Pinky"):
            ids = FINGER_JOINTS[finger_name]
            mcp_id, pip_id, dip_id, tip_id = ids[0], ids[1], ids[2], ids[3]

            p_mcp = joints_3d[mcp_id]
            p_pip = joints_3d[pip_id]
            p_dip = joints_3d[dip_id]
            p_tip = joints_3d[tip_id]

            # MCP flexion: angle between Wrist->MCP and MCP->PIP
            mcp_angle = cls.joint_flexion_angle(wrist, p_mcp, p_pip)
            # PIP flexion: angle between MCP->PIP and PIP->DIP
            pip_angle = cls.joint_flexion_angle(p_mcp, p_pip, p_dip)
            # DIP flexion: angle between PIP->DIP and DIP->Tip
            dip_angle = cls.joint_flexion_angle(p_pip, p_dip, p_tip)

            total_flexion = mcp_angle + pip_angle + dip_angle
            # Finger is extended if PIP flexion < 35° and DIP flexion < 35°
            is_extended = pip_angle < 35.0 and dip_angle < 35.0

            angles_map[finger_name] = FingerAngles(
                finger_name=finger_name,
                mcp_flexion_deg=round(mcp_angle, 1),
                pip_flexion_deg=round(pip_angle, 1),
                dip_flexion_deg=round(dip_angle, 1),
                total_flexion_deg=round(total_flexion, 1),
                is_extended=is_extended,
            )

        # 2. Thumb Digits (CMC 1, MCP 2, IP 3, TIP 4)
        p_cmc = joints_3d[JOINT_THUMB_CMC]
        p_tmcp = joints_3d[JOINT_THUMB_MCP]
        p_tip = joints_3d[JOINT_THUMB_IP]
        p_ttip = joints_3d[JOINT_THUMB_TIP]

        thumb_mcp_flex = cls.joint_flexion_angle(p_cmc, p_tmcp, p_tip)
        thumb_ip_flex = cls.joint_flexion_angle(p_tmcp, p_tip, p_ttip)
        thumb_total = thumb_mcp_flex + thumb_ip_flex
        thumb_is_extended = thumb_ip_flex < 32.0

        angles_map["Thumb"] = FingerAngles(
            finger_name="Thumb",
            mcp_flexion_deg=round(thumb_mcp_flex, 1),
            pip_flexion_deg=round(thumb_ip_flex, 1),
            dip_flexion_deg=0.0,
            total_flexion_deg=round(thumb_total, 1),
            is_extended=thumb_is_extended,
        )

        # 3. Inter-Finger Abduction (Spread Angles)
        v_index = cls._vector(joints_3d[JOINT_INDEX_MCP], joints_3d[8])
        v_middle = cls._vector(joints_3d[JOINT_MIDDLE_MCP], joints_3d[12])
        v_ring = cls._vector(joints_3d[JOINT_RING_MCP], joints_3d[16])
        v_pinky = cls._vector(joints_3d[JOINT_PINKY_MCP], joints_3d[20])
        v_thumb = cls._vector(joints_3d[JOINT_THUMB_MCP], joints_3d[JOINT_THUMB_TIP])

        spread_im = cls.angle_between_vectors(v_index, v_middle)
        spread_mr = cls.angle_between_vectors(v_middle, v_ring)
        spread_rp = cls.angle_between_vectors(v_ring, v_pinky)
        spread_thumb_idx = cls.angle_between_vectors(v_thumb, v_index)

        angles_map["Index"].abduction_deg = round(spread_im, 1)
        angles_map["Middle"].abduction_deg = round(spread_mr, 1)
        angles_map["Ring"].abduction_deg = round(spread_rp, 1)
        angles_map["Thumb"].abduction_deg = round(spread_thumb_idx, 1)

        # 4. Thumb Opposition: angle between Thumb vector and palm lateral axis (Index MCP to Pinky MCP)
        v_palm_lateral = cls._vector(joints_3d[JOINT_INDEX_MCP], joints_3d[JOINT_PINKY_MCP])
        thumb_opposition = cls.angle_between_vectors(v_thumb, v_palm_lateral)

        return HandJointAngles(
            fingers=angles_map,
            thumb_opposition_deg=round(thumb_opposition, 1),
            thumb_abduction_deg=round(spread_thumb_idx, 1),
            index_middle_spread_deg=round(spread_im, 1),
            middle_ring_spread_deg=round(spread_mr, 1),
            ring_pinky_spread_deg=round(spread_rp, 1),
        )

