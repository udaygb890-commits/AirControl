"""
AIRCONTROL — 3D HAND TRACKING AND ARTICULATED KINEMATICS
Package providing 3D articulated skeleton representations, joint flexion angles,
palm orientation, spatial metrics, One-Euro temporal filtering, calibration,
and real-time 3D wireframe rendering.
"""

from hand3d.skeleton import (
    Bone3D,
    FINGER_JOINTS,
    FINGERTIP_IDS,
    HandSkeleton,
    JOINT_NAMES,
    JOINT_WRIST,
)
from hand3d.joint_angles import (
    FingerAngles,
    HandJointAngles,
    JointAngleCalculator,
)
from hand3d.hand_orientation import (
    HandOrientation,
    OrientationEstimator,
    PalmFacing,
)
from hand3d.spatial_features import (
    SpatialFeatureExtractor,
    SpatialFeatures,
)
from hand3d.hand_filter import (
    HandFilter3D,
    OneEuroFilter1D,
    OneEuroFilter3D,
)
from hand3d.calibration import (
    CalibrationProfile,
    HandCalibrator,
)
from hand3d.hand_model import (
    Hand3DProcessor,
    HandModel3D,
)
from hand3d.renderer_3d import (
    SkeletonRenderer3D,
)

__all__ = [
    "Bone3D",
    "FINGER_JOINTS",
    "FINGERTIP_IDS",
    "HandSkeleton",
    "JOINT_NAMES",
    "JOINT_WRIST",
    "FingerAngles",
    "HandJointAngles",
    "JointAngleCalculator",
    "HandOrientation",
    "OrientationEstimator",
    "PalmFacing",
    "SpatialFeatureExtractor",
    "SpatialFeatures",
    "HandFilter3D",
    "OneEuroFilter1D",
    "OneEuroFilter3D",
    "CalibrationProfile",
    "HandCalibrator",
    "Hand3DProcessor",
    "HandModel3D",
    "SkeletonRenderer3D",
]

