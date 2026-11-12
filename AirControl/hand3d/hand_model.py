"""
AIRCONTROL — 3D ARTICULATED HAND MODEL
Consolidates 21 articulated joints, 20 bones, 3D joint flexion angles,
palm orientation, canonical scale-invariant features, and temporal filtering.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hand3d.joint_angles import FingerAngles, HandJointAngles, JointAngleCalculator
from hand3d.hand_filter import HandFilter3D
from hand3d.hand_orientation import HandOrientation, OrientationEstimator, PalmFacing
from hand3d.skeleton import (
    Bone3D,
    FINGERTIP_IDS,
    HandSkeleton,
    JOINT_NAMES,
    JOINT_WRIST,
)
from hand3d.spatial_features import SpatialFeatureExtractor, SpatialFeatures


@dataclass
class HandModel3D:
    """
    Complete articulated 3D hand representation.
    """
    handedness: str = "Right"
    confidence: float = 0.0

    # 3D Coordinates (21 landmarks)
    joints_raw: List[Tuple[float, float, float]] = field(default_factory=list)
    joints_filtered: List[Tuple[float, float, float]] = field(default_factory=list)
    joints_canonical: List[Tuple[float, float, float]] = field(default_factory=list)
    joints_norm: List[Tuple[float, float, float]] = field(default_factory=list)

    # Articulated Skeleton Structure
    bones: List[Bone3D] = field(default_factory=list)

    # Biomechanical Kinematics
    angles: HandJointAngles = field(default_factory=HandJointAngles)
    orientation: HandOrientation = field(default_factory=HandOrientation)
    spatial: SpatialFeatures = field(default_factory=SpatialFeatures)

    palm_scale: float = 1.0
    is_valid: bool = False
    timestamp: float = 0.0

    def is_finger_straight(self, finger_name: str, threshold_deg: float = 45.0) -> bool:
        """
        Check if a digit is extended straight based on cumulative 3D flexion angle.
        """
        if finger_name not in self.angles.fingers:
            return False
        return self.angles.fingers[finger_name].total_flexion_deg < threshold_deg

    def is_finger_curled(self, finger_name: str, threshold_deg: float = 95.0) -> bool:
        """
        Check if a digit is curled/flexed into the palm based on cumulative 3D flexion angle.
        """
        if finger_name not in self.angles.fingers:
            return False
        return self.angles.fingers[finger_name].total_flexion_deg > threshold_deg

    def get_pinch_distance(self, finger_name: str = "Index") -> float:
        """
        Get canonical 3D distance between thumb tip and specified finger tip.
        """
        if finger_name == "Index":
            return self.spatial.thumb_index_dist
        elif finger_name == "Middle":
            return self.spatial.thumb_middle_dist
        elif finger_name == "Ring":
            return self.spatial.thumb_ring_dist
        elif finger_name == "Pinky":
            return self.spatial.thumb_pinky_dist
        return 999.0

    def is_palm_facing_camera(self) -> bool:
        """True if palm surface is oriented toward the camera."""
        return self.orientation.facing == PalmFacing.FACING_CAMERA

    def is_palm_facing_away(self) -> bool:
        """True if back of hand faces camera (palm points into screen)."""
        return self.orientation.facing == PalmFacing.FACING_AWAY

    def is_palm_facing_up(self) -> bool:
        return self.orientation.facing == PalmFacing.FACING_UP

    def is_palm_facing_down(self) -> bool:
        return self.orientation.facing == PalmFacing.FACING_DOWN

    def get_joint(self, joint_id: int) -> Tuple[float, float, float]:
        """Get 3D filtered position of joint by index (0-20)."""
        if 0 <= joint_id < len(self.joints_filtered):
            return self.joints_filtered[joint_id]
        return (0.0, 0.0, 0.0)


class Hand3DProcessor:
    """
    Stateful processor that maintains temporal filters and estimators
    for a hand stream to generate smooth HandModel3D frames.
    """

    def __init__(self, handedness: str = "Right"):
        self.handedness = handedness
        self.is_right_hand = (handedness.lower() == "right")

        self.skeleton = HandSkeleton()
        self.filter = HandFilter3D()
        self.orientation_estimator = OrientationEstimator(self.is_right_hand)
        self.spatial_extractor = SpatialFeatureExtractor()

    def update(
        self,
        joints_3d: List[Tuple[float, float, float]],
        joints_norm: Optional[List[Tuple[float, float, float]]] = None,
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> HandModel3D:
        """
        Process new 3D landmarks into a complete HandModel3D instance.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        if len(joints_3d) < 21:
            self.filter.on_tracking_lost()
            return HandModel3D(handedness=self.handedness, is_valid=False, timestamp=timestamp)

        # Sanitize any NaN or Inf coordinates
        clean_joints: List[Tuple[float, float, float]] = []
        for p in joints_3d:
            clean_joints.append((
                0.0 if (math.isnan(p[0]) or math.isinf(p[0])) else float(p[0]),
                0.0 if (math.isnan(p[1]) or math.isinf(p[1])) else float(p[1]),
                0.0 if (math.isnan(p[2]) or math.isinf(p[2])) else float(p[2]),
            ))

        # 1. Temporal Low-Pass / One-Euro Filtering
        filtered_joints = self.filter.filter_joints(clean_joints, timestamp)

        # 2. Articulated Bones
        bones = self.skeleton.calculate_bones(filtered_joints)

        # 3. 3D Joint Angles & Flexion
        angles = JointAngleCalculator.calculate_hand_angles(filtered_joints)

        # 4. 3D Hand Orientation & Palm Normal
        orientation = self.orientation_estimator.estimate(filtered_joints, timestamp)

        # 5. Canonical Invariant Frame & Spatial Metrics
        spatial = self.spatial_extractor.extract(filtered_joints, orientation.palm_normal, timestamp)

        return HandModel3D(
            handedness=self.handedness,
            confidence=confidence,
            joints_raw=list(joints_3d),
            joints_filtered=filtered_joints,
            joints_canonical=spatial.canonical_joints,
            joints_norm=list(joints_norm) if joints_norm else list(joints_3d),
            bones=bones,
            angles=angles,
            orientation=orientation,
            spatial=spatial,
            palm_scale=spatial.palm_scale,
            is_valid=True,
            timestamp=timestamp,
        )

    def reset(self) -> None:
        """Reset internal temporal states."""
        self.filter.reset()
        self.orientation_estimator.reset()
        self.spatial_extractor.reset()
