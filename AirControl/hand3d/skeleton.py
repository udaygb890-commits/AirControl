"""
AIRCONTROL — 3D HAND SKELETON MODULE
Defines the anatomical 21-joint skeleton, bone segments, joint hierarchy,
and kinematic parent-child relationships for articulated 3D hand tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Landmark IDs matching MediaPipe Hand 21-point topology
JOINT_WRIST = 0

JOINT_THUMB_CMC = 1
JOINT_THUMB_MCP = 2
JOINT_THUMB_IP = 3
JOINT_THUMB_TIP = 4

JOINT_INDEX_MCP = 5
JOINT_INDEX_PIP = 6
JOINT_INDEX_DIP = 7
JOINT_INDEX_TIP = 8

JOINT_MIDDLE_MCP = 9
JOINT_MIDDLE_PIP = 10
JOINT_MIDDLE_DIP = 11
JOINT_MIDDLE_TIP = 12

JOINT_RING_MCP = 13
JOINT_RING_PIP = 14
JOINT_RING_DIP = 15
JOINT_RING_TIP = 16

JOINT_PINKY_MCP = 17
JOINT_PINKY_PIP = 18
JOINT_PINKY_DIP = 19
JOINT_PINKY_TIP = 20

# Canonical Anatomical Names
JOINT_NAMES: Dict[int, str] = {
    0: "WRIST",
    1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
    9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
    13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
}

# Finger to Joint Index Groups
FINGER_JOINTS: Dict[str, List[int]] = {
    "Thumb": [JOINT_THUMB_CMC, JOINT_THUMB_MCP, JOINT_THUMB_IP, JOINT_THUMB_TIP],
    "Index": [JOINT_INDEX_MCP, JOINT_INDEX_PIP, JOINT_INDEX_DIP, JOINT_INDEX_TIP],
    "Middle": [JOINT_MIDDLE_MCP, JOINT_MIDDLE_PIP, JOINT_MIDDLE_DIP, JOINT_MIDDLE_TIP],
    "Ring": [JOINT_RING_MCP, JOINT_RING_PIP, JOINT_RING_DIP, JOINT_RING_TIP],
    "Pinky": [JOINT_PINKY_MCP, JOINT_PINKY_PIP, JOINT_PINKY_DIP, JOINT_PINKY_TIP],
}

FINGERTIP_IDS: Dict[str, int] = {
    "Thumb": JOINT_THUMB_TIP,
    "Index": JOINT_INDEX_TIP,
    "Middle": JOINT_MIDDLE_TIP,
    "Ring": JOINT_RING_TIP,
    "Pinky": JOINT_PINKY_TIP,
}

MCP_IDS: Dict[str, int] = {
    "Thumb": JOINT_THUMB_MCP,
    "Index": JOINT_INDEX_MCP,
    "Middle": JOINT_MIDDLE_MCP,
    "Ring": JOINT_RING_MCP,
    "Pinky": JOINT_PINKY_MCP,
}

# Anatomical Bone Segments: (parent_joint_id, child_joint_id, bone_name, finger_name)
BONE_DEFINITIONS: List[Tuple[int, int, str, str]] = [
    # Palm Base Connections
    (0, 1, "WRIST_THUMB_CMC", "Palm"),
    (0, 5, "WRIST_INDEX_MCP", "Palm"),
    (0, 9, "WRIST_MIDDLE_MCP", "Palm"),
    (0, 13, "WRIST_RING_MCP", "Palm"),
    (0, 17, "WRIST_PINKY_MCP", "Palm"),
    (5, 9, "INDEX_MIDDLE_MCP", "Palm"),
    (9, 13, "MIDDLE_RING_MCP", "Palm"),
    (13, 17, "RING_PINKY_MCP", "Palm"),

    # Thumb Phalanx Chain
    (1, 2, "THUMB_METACARPAL", "Thumb"),
    (2, 3, "THUMB_PROXIMAL", "Thumb"),
    (3, 4, "THUMB_DISTAL", "Thumb"),

    # Index Phalanx Chain
    (5, 6, "INDEX_PROXIMAL", "Index"),
    (6, 7, "INDEX_INTERMEDIATE", "Index"),
    (7, 8, "INDEX_DISTAL", "Index"),

    # Middle Phalanx Chain
    (9, 10, "MIDDLE_PROXIMAL", "Middle"),
    (10, 11, "MIDDLE_INTERMEDIATE", "Middle"),
    (11, 12, "MIDDLE_DISTAL", "Middle"),

    # Ring Phalanx Chain
    (13, 14, "RING_PROXIMAL", "Ring"),
    (14, 15, "RING_INTERMEDIATE", "Ring"),
    (15, 16, "RING_DISTAL", "Ring"),

    # Pinky Phalanx Chain
    (17, 18, "PINKY_PROXIMAL", "Pinky"),
    (18, 19, "PINKY_INTERMEDIATE", "Pinky"),
    (19, 20, "PINKY_DISTAL", "Pinky"),
]

# Kinematic Parent Joint Hierarchy (Child -> Parent)
PARENT_JOINT_MAP: Dict[int, int] = {
    1: 0, 2: 1, 3: 2, 4: 3,
    5: 0, 6: 5, 7: 6, 8: 7,
    9: 0, 10: 9, 11: 10, 12: 11,
    13: 0, 14: 13, 15: 14, 16: 15,
    17: 0, 18: 17, 19: 18, 20: 19,
}


@dataclass
class Bone3D:
    """Represents an articulated anatomical bone connecting two joints."""
    start_id: int
    end_id: int
    name: str
    finger_name: str
    length: float = 0.0
    direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class HandSkeleton:
    """
    Manages anatomical bone connections and joint kinematic hierarchy.
    Computes 3D bone vectors, lengths, and joint relationships.
    """

    def __init__(self):
        self.bone_definitions = list(BONE_DEFINITIONS)
        self.parent_map = dict(PARENT_JOINT_MAP)

    def calculate_bones(
        self,
        joints_3d: List[Tuple[float, float, float]],
    ) -> List[Bone3D]:
        """
        Calculate 3D bone lengths and unit direction vectors from 21 joint coordinates.

        Args:
            joints_3d: List of 21 3D coordinates (x, y, z).

        Returns:
            List of Bone3D instances.
        """
        if len(joints_3d) < 21:
            return []

        bones: List[Bone3D] = []
        for start_id, end_id, name, finger in self.bone_definitions:
            p1 = joints_3d[start_id]
            p2 = joints_3d[end_id]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)

            if length > 1e-6:
                inv = 1.0 / length
                direction = (dx * inv, dy * inv, dz * inv)
            else:
                direction = (0.0, 0.0, 0.0)

            bones.append(
                Bone3D(
                    start_id=start_id,
                    end_id=end_id,
                    name=name,
                    finger_name=finger,
                    length=length,
                    direction=direction,
                )
            )

        return bones

    @staticmethod
    def get_finger_bones(bones: List[Bone3D], finger_name: str) -> List[Bone3D]:
        """Filter bones belonging to a specific finger or the palm."""
        return [b for b in bones if b.finger_name == finger_name]

