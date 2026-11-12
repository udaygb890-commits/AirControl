"""
AIRCONTROL — GESTURE CLASSIFIER MODULE (STAGE 3)
Robust geometric classification of static hand poses from 21 MediaPipe 3D landmarks.
Scale-invariant, translation-invariant, and rotation-aware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from vision.hand_tracker import HandData


class StaticGesture(str, Enum):
    """Canonical catalog of recognized static hand poses."""
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    POINTING = "POINTING"
    VICTORY = "VICTORY"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"
    ROCK_ON = "ROCK_ON"
    SPIDERMAN = "SPIDERMAN"
    PINCH_OK = "PINCH_OK"
    THREE_FINGERS = "THREE_FINGERS"
    FOUR_FINGERS = "FOUR_FINGERS"
    CALL_ME = "CALL_ME"
    UNKNOWN = "UNKNOWN"


@dataclass
class GestureClassification:
    """Detailed structural classification result for a single hand pose."""
    name: str                              # Member of StaticGesture
    confidence: float                      # 0.0 to 1.0 confidence score
    finger_curls: Dict[str, float]         # Finger name -> curl ratio (0.0=straight, 1.0=fully curled)
    finger_extensions: Dict[str, bool]     # Finger name -> is_extended (True/False)
    palm_direction: str                    # "FACING_CAMERA", "AWAY_FROM_CAMERA", "SIDEWAYS"
    hand_orientation: str                  # "UP", "DOWN", "LEFT", "RIGHT"
    hand_angle_deg: float                  # Planar orientation angle in degrees (-180 to +180)
    palm_scale: float                      # Normalized scale reference
    description: str = ""                  # Human-readable tactical description


class GestureClassifier:
    """
    Analyzes 21 3D hand landmarks to extract anatomical joint angles,
    finger straightness ratios, thumb vectors, and spatial palm orientation
    to classify static hand poses with high precision.
    """

    FINGER_JOINTS = {
        "Index": (5, 6, 7, 8),      # MCP, PIP, DIP, TIP
        "Middle": (9, 10, 11, 12),
        "Ring": (13, 14, 15, 16),
        "Pinky": (17, 18, 19, 20),
    }

    def __init__(self):
        pass

    @staticmethod
    def _dist3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Euclidean 3D distance between two landmarks."""
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2
        )

    def _compute_finger_curl(
        self,
        lms: List[Tuple[float, float, float]],
        joints: Tuple[int, int, int, int],
        palm_scale: float,
    ) -> float:
        """
        Compute continuous finger curl metric in [0.0, 1.0].
        0.0 = completely extended / straight,
        1.0 = tightly curled into palm.
        """
        mcp, pip, dip, tip = joints
        p_mcp, p_pip, p_dip, p_tip = lms[mcp], lms[pip], lms[dip], lms[tip]

        # Total anatomical length along finger segments
        seg_len = (
            self._dist3d(p_mcp, p_pip) +
            self._dist3d(p_pip, p_dip) +
            self._dist3d(p_dip, p_tip)
        )
        if seg_len < 1e-4:
            return 1.0

        # Straight-line distance from MCP to TIP
        mcp_to_tip = self._dist3d(p_mcp, p_tip)
        straightness = mcp_to_tip / seg_len

        # For fully extended finger: straightness ~ 0.95 - 1.0
        # For fully curled finger: straightness ~ 0.30 - 0.45
        curl = (0.90 - straightness) / (0.90 - 0.40)
        return float(np.clip(curl, 0.0, 1.0))

    def _compute_thumb_curl(
        self,
        lms: List[Tuple[float, float, float]],
        palm_scale: float,
    ) -> float:
        """
        Compute thumb curl ratio considering both segment straightness
        and thumb tip proximity to pinky/middle MCP joints.
        """
        p_cmc = lms[1]
        p_mcp = lms[2]
        p_ip = lms[3]
        p_tip = lms[4]

        seg_len = (
            self._dist3d(p_cmc, p_mcp) +
            self._dist3d(p_mcp, p_ip) +
            self._dist3d(p_ip, p_tip)
        )
        if seg_len < 1e-4:
            return 1.0

        cmc_to_tip = self._dist3d(p_cmc, p_tip)
        straightness = cmc_to_tip / seg_len

        # Distance to base of other fingers (palm tuck)
        dist_to_middle_mcp = self._dist3d(p_tip, lms[9]) / max(0.01, palm_scale)
        dist_to_pinky_mcp = self._dist3d(p_tip, lms[17]) / max(0.01, palm_scale)
        dist_to_wrist = self._dist3d(p_tip, lms[0]) / max(0.01, palm_scale)

        # Base curl from segment straightness
        base_curl = (0.92 - straightness) / (0.92 - 0.45)

        # If tip is tucked close to palm base and not extended away from wrist, thumb is curled
        if dist_to_wrist < 1.05 and (dist_to_pinky_mcp < 0.65 or dist_to_middle_mcp < 0.60):
            base_curl = max(base_curl, 0.75)

        return float(np.clip(base_curl, 0.0, 1.0))

    def _compute_palm_orientation(
        self,
        lms: List[Tuple[float, float, float]],
        handedness: str,
    ) -> Tuple[str, str, float]:
        """
        Determine palm facing direction, hand tilt orientation, and planar angle.

        Returns:
            (palm_direction, hand_orientation, angle_deg)
        """
        wrist = lms[0]
        middle_mcp = lms[9]
        index_mcp = lms[5]
        pinky_mcp = lms[17]

        # 1. Planar vector from Wrist to Middle MCP (Direction hand is pointing)
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]  # Note: y points DOWN in image coordinates
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Classify primary pointing direction:
        # Image coordinates: y increases downward.
        # UP: -135 to -45 deg
        # RIGHT: -45 to 45 deg
        # DOWN: 45 to 135 deg
        # LEFT: > 135 or < -135 deg
        if -135.0 <= angle_deg < -45.0:
            orientation = "UP"
        elif -45.0 <= angle_deg < 45.0:
            orientation = "RIGHT"
        elif 45.0 <= angle_deg < 135.0:
            orientation = "DOWN"
        else:
            orientation = "LEFT"

        # 2. Palm Normal Vector via Cross Product
        # Vector v1: Wrist -> Index MCP
        # Vector v2: Wrist -> Pinky MCP
        v1 = np.array([index_mcp[0] - wrist[0], index_mcp[1] - wrist[1], index_mcp[2] - wrist[2]])
        v2 = np.array([pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1], pinky_mcp[2] - wrist[2]])
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)

        if norm_len > 1e-5:
            normal = normal / norm_len

        # MediaPipe z-axis: points towards camera (negative z is deeper into scene)
        # Taking handedness into account:
        nz = normal[2]
        if handedness == "Right":
            facing = "FACING_CAMERA" if nz > 0.15 else ("AWAY_FROM_CAMERA" if nz < -0.15 else "SIDEWAYS")
        else:
            facing = "FACING_CAMERA" if nz < -0.15 else ("AWAY_FROM_CAMERA" if nz > 0.15 else "SIDEWAYS")

        return facing, orientation, angle_deg

    def classify(self, hand: HandData) -> GestureClassification:
        """
        Classify static hand gesture from HandData landmarks.

        Args:
            hand: Structured HandData from HandTracker.

        Returns:
            GestureClassification with predicted gesture, confidence, and metrics.
        """
        lms = hand.landmarks_norm
        if len(lms) < 21:
            return GestureClassification(
                name=StaticGesture.UNKNOWN.value,
                confidence=0.0,
                finger_curls={},
                finger_extensions={},
                palm_direction="SIDEWAYS",
                hand_orientation="UP",
                hand_angle_deg=0.0,
                palm_scale=0.1,
                description="Insufficient landmark coordinates",
            )

        # Scale reference: Wrist (0) to Middle MCP (9)
        palm_scale = max(0.01, self._dist3d(lms[0], lms[9]))

        # 1. Compute Curls & Extensions
        curls: Dict[str, float] = {}
        for fname, joints in self.FINGER_JOINTS.items():
            curls[fname] = self._compute_finger_curl(lms, joints, palm_scale)
        curls["Thumb"] = self._compute_thumb_curl(lms, palm_scale)

        extensions: Dict[str, bool] = {
            fname: curl < 0.45 for fname, curl in curls.items()
        }

        # 2. Palm Spatial Orientation
        palm_dir, hand_orient, angle_deg = self._compute_palm_orientation(lms, hand.handedness)

        # 3. Inter-landmark Metric Measurements
        thumb_tip = lms[4]
        index_tip = lms[8]
        middle_tip = lms[12]
        ring_tip = lms[16]
        pinky_tip = lms[20]

        # Thumb to fingertip distances (normalized by palm scale)
        d_thumb_index = self._dist3d(thumb_tip, index_tip) / palm_scale
        d_thumb_middle = self._dist3d(thumb_tip, middle_tip) / palm_scale
        d_index_middle = self._dist3d(index_tip, middle_tip) / palm_scale

        # Thumb vector relative to Wrist
        thumb_dx = thumb_tip[0] - lms[0][0]
        thumb_dy = thumb_tip[1] - lms[0][1]  # negative = upward

        thumb_upward = thumb_dy < -0.30 * palm_scale and abs(thumb_dy) > 0.9 * abs(thumb_dx)
        thumb_downward = thumb_dy > 0.30 * palm_scale and abs(thumb_dy) > 0.9 * abs(thumb_dx)

        ext_count = sum(extensions.values())
        curled_count = sum(1 for c in curls.values() if c > 0.55)

        # 4. Pattern Matching / Pose Evaluation
        gesture_name = StaticGesture.UNKNOWN.value
        confidence = 0.50
        desc = "Ambiguous hand position"

        # ----------------------------------------------------------------------
        # A. OPEN PALM / STOP
        # All 5 fingers extended outward
        # ----------------------------------------------------------------------
        if all(extensions[f] for f in ["Index", "Middle", "Ring", "Pinky"]) and curls["Thumb"] < 0.50:
            gesture_name = StaticGesture.OPEN_PALM.value
            avg_curl = sum(curls.values()) / 5.0
            confidence = float(np.clip(1.0 - avg_curl * 0.8, 0.70, 0.99))
            desc = "Open Palm — All 5 fingers fully extended"

        # ----------------------------------------------------------------------
        # B. THUMBS UP / THUMBS DOWN
        # 4 fingers curled, thumb extended pointing UP or DOWN
        # ----------------------------------------------------------------------
        elif all(curls[f] > 0.55 for f in ["Index", "Middle", "Ring", "Pinky"]) and curls["Thumb"] < 0.45:
            tip_mcp_dy = thumb_tip[1] - lms[2][1]
            if thumb_downward or (tip_mcp_dy > 0.15 * palm_scale):
                gesture_name = StaticGesture.THUMBS_DOWN.value
                confidence = float(np.clip(0.85 + (1.0 - curls["Thumb"]) * 0.14, 0.75, 0.99))
                desc = "Thumbs Down — Negative gesture"
            elif thumb_upward or (tip_mcp_dy < -0.15 * palm_scale):
                gesture_name = StaticGesture.THUMBS_UP.value
                confidence = float(np.clip(0.85 + (1.0 - curls["Thumb"]) * 0.14, 0.75, 0.99))
                desc = "Thumbs Up — Affirmative gesture"
            else:
                gesture_name = StaticGesture.THUMBS_UP.value if hand_orient == "UP" else StaticGesture.THUMBS_DOWN.value
                confidence = 0.70
                desc = "Thumb Extended"

        # ----------------------------------------------------------------------
        # C. FIST / CLUTCH
        # All 4 main fingers curled tight, thumb folded
        # ----------------------------------------------------------------------
        elif all(curls[f] > 0.58 for f in ["Index", "Middle", "Ring", "Pinky"]) and curls["Thumb"] > 0.45:
            gesture_name = StaticGesture.FIST.value
            avg_curl = (curls["Index"] + curls["Middle"] + curls["Ring"] + curls["Pinky"]) / 4.0
            confidence = float(np.clip(avg_curl, 0.70, 0.99))
            desc = "Fist — All fingers curled tight"

        # ----------------------------------------------------------------------
        # D. POINTING
        # Index extended, Middle, Ring, Pinky curled
        # ----------------------------------------------------------------------
        elif extensions["Index"] and all(curls[f] > 0.55 for f in ["Middle", "Ring", "Pinky"]):
            gesture_name = StaticGesture.POINTING.value
            conf = (1.0 - curls["Index"]) * 0.5 + (curls["Middle"] + curls["Ring"] + curls["Pinky"]) / 3.0 * 0.5
            confidence = float(np.clip(conf, 0.75, 0.99))
            desc = "Pointing — Index finger extended"

        # ----------------------------------------------------------------------
        # E. VICTORY / PEACE
        # Index and Middle extended in V-shape, Ring & Pinky curled
        # ----------------------------------------------------------------------
        elif (
            extensions["Index"] and
            extensions["Middle"] and
            curls["Ring"] > 0.55 and
            curls["Pinky"] > 0.55 and
            d_index_middle > 0.22
        ):
            gesture_name = StaticGesture.VICTORY.value
            conf = ((1.0 - curls["Index"]) + (1.0 - curls["Middle"]) + curls["Ring"] + curls["Pinky"]) / 4.0
            confidence = float(np.clip(conf, 0.75, 0.99))
            desc = "Victory / Peace — Index and Middle fingers in V"

        # ----------------------------------------------------------------------
        # F. ROCK ON / HORNS
        # Index and Pinky extended, Middle and Ring curled
        # ----------------------------------------------------------------------
        elif (
            extensions["Index"] and
            extensions["Pinky"] and
            curls["Middle"] > 0.55 and
            curls["Ring"] > 0.55 and
            curls["Thumb"] > 0.35
        ):
            gesture_name = StaticGesture.ROCK_ON.value
            conf = ((1.0 - curls["Index"]) + (1.0 - curls["Pinky"]) + curls["Middle"] + curls["Ring"]) / 4.0
            confidence = float(np.clip(conf, 0.75, 0.99))
            desc = "Rock On / Horns — Index and Pinky extended"

        # ----------------------------------------------------------------------
        # G. SPIDERMAN / I LOVE YOU
        # Thumb, Index, Pinky extended, Middle and Ring curled
        # ----------------------------------------------------------------------
        elif (
            extensions["Thumb"] and
            extensions["Index"] and
            extensions["Pinky"] and
            curls["Middle"] > 0.55 and
            curls["Ring"] > 0.55
        ):
            gesture_name = StaticGesture.SPIDERMAN.value
            conf = ((1.0 - curls["Index"]) + (1.0 - curls["Pinky"]) + (1.0 - curls["Thumb"])) / 3.0
            confidence = float(np.clip(conf, 0.75, 0.99))
            desc = "Spiderman / Web-Shooter — Thumb, Index, Pinky extended"

        # ----------------------------------------------------------------------
        # H. PINCH OK
        # Thumb and Index tips touching in a ring, other 3 fingers extended
        # ----------------------------------------------------------------------
        elif (
            d_thumb_index < 0.30 and
            extensions["Middle"] and
            extensions["Ring"] and
            extensions["Pinky"]
        ):
            gesture_name = StaticGesture.PINCH_OK.value
            confidence = float(np.clip(1.0 - d_thumb_index * 1.5, 0.75, 0.98))
            desc = "OK Sign — Thumb and Index tips touching"

        # ----------------------------------------------------------------------
        # I. CALL ME
        # Thumb and Pinky extended, Middle three curled
        # ----------------------------------------------------------------------
        elif (
            extensions["Thumb"] and
            extensions["Pinky"] and
            curls["Index"] > 0.55 and
            curls["Middle"] > 0.55 and
            curls["Ring"] > 0.55
        ):
            gesture_name = StaticGesture.CALL_ME.value
            confidence = float(np.clip(((1.0 - curls["Thumb"]) + (1.0 - curls["Pinky"])) / 2.0, 0.75, 0.98))
            desc = "Call Me — Thumb and Pinky extended outward"

        # ----------------------------------------------------------------------
        # J. THREE FINGERS
        # Index, Middle, Ring extended, Pinky curled
        # ----------------------------------------------------------------------
        elif (
            extensions["Index"] and
            extensions["Middle"] and
            extensions["Ring"] and
            curls["Pinky"] > 0.55
        ):
            gesture_name = StaticGesture.THREE_FINGERS.value
            confidence = 0.82
            desc = "Three Fingers — Index, Middle, Ring extended"

        # ----------------------------------------------------------------------
        # K. FOUR FINGERS
        # All 4 fingers extended, Thumb curled across palm
        # ----------------------------------------------------------------------
        elif (
            all(extensions[f] for f in ["Index", "Middle", "Ring", "Pinky"]) and
            curls["Thumb"] > 0.55
        ):
            gesture_name = StaticGesture.FOUR_FINGERS.value
            confidence = 0.85
            desc = "Four Fingers — 4 fingers up, thumb tucked"

        return GestureClassification(
            name=gesture_name,
            confidence=confidence,
            finger_curls=curls,
            finger_extensions=extensions,
            palm_direction=palm_dir,
            hand_orientation=hand_orient,
            hand_angle_deg=round(angle_deg, 1),
            palm_scale=round(palm_scale, 4),
            description=desc,
        )
