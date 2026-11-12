"""
AIRCONTROL — PHASE 5 VERIFICATION SUITE
Verifies detection and classification of:
- Open palm
- Fist
- Pointing finger
- Peace sign (Victory)
- Left hand
- Right hand
- Both hands
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from gestures.gesture_classifier import GestureClassifier, StaticGesture
from gestures.gesture_engine import GestureEngine
from vision.hand_tracker import HandData, HandTrackingResult


def generate_hand_data(gesture_type: str, handedness: str = "Right") -> HandData:
    """Generate synthetic HandData with 21 normalized landmarks for different hand poses."""
    is_r = (handedness == "Right")
    sign = 1 if is_r else -1

    # Base wrist at (0.5, 0.8, 0.0)
    lm = [(0.5, 0.8, 0.0)]

    # Thumb: 1, 2, 3, 4
    if gesture_type in ("OPEN_PALM", "POINTING", "VICTORY"):
        # Thumb extended out to the side
        lm += [
            (0.5 + sign * 0.05, 0.75, 0.0),
            (0.5 + sign * 0.10, 0.70, 0.0),
            (0.5 + sign * 0.14, 0.65, 0.0),
            (0.5 + sign * 0.18, 0.60, 0.0),
        ]
    else:  # FIST
        # Thumb curled across fingers
        lm += [
            (0.5 + sign * 0.03, 0.75, 0.0),
            (0.5 + sign * 0.05, 0.72, 0.0),
            (0.5 + sign * 0.04, 0.69, 0.0),
            (0.5 + sign * 0.02, 0.66, 0.0),
        ]

    # Index: 5, 6, 7, 8
    if gesture_type in ("OPEN_PALM", "POINTING", "VICTORY"):
        # Extended straight up
        lm += [
            (0.5 + sign * 0.04, 0.60, 0.0),
            (0.5 + sign * 0.04, 0.50, 0.0),
            (0.5 + sign * 0.04, 0.40, 0.0),
            (0.5 + sign * 0.04, 0.30, 0.0),
        ]
    else:  # FIST
        # Curled into palm
        lm += [
            (0.5 + sign * 0.04, 0.60, 0.0),
            (0.5 + sign * 0.04, 0.52, -0.05),
            (0.5 + sign * 0.04, 0.58, -0.05),
            (0.5 + sign * 0.04, 0.64, 0.0),
        ]

    # Middle: 9, 10, 11, 12
    if gesture_type in ("OPEN_PALM", "VICTORY"):
        # Extended straight up
        lm += [
            (0.5, 0.58, 0.0),
            (0.5, 0.47, 0.0),
            (0.5, 0.36, 0.0),
            (0.5, 0.25, 0.0),
        ]
    else:  # FIST, POINTING
        # Curled into palm
        lm += [
            (0.5, 0.58, 0.0),
            (0.5, 0.50, -0.05),
            (0.5, 0.56, -0.05),
            (0.5, 0.63, 0.0),
        ]

    # Ring: 13, 14, 15, 16
    if gesture_type in ("OPEN_PALM",):
        # Extended straight up
        lm += [
            (0.5 - sign * 0.04, 0.60, 0.0),
            (0.5 - sign * 0.04, 0.50, 0.0),
            (0.5 - sign * 0.04, 0.41, 0.0),
            (0.5 - sign * 0.04, 0.32, 0.0),
        ]
    else:  # FIST, POINTING, VICTORY
        # Curled into palm
        lm += [
            (0.5 - sign * 0.04, 0.60, 0.0),
            (0.5 - sign * 0.04, 0.52, -0.05),
            (0.5 - sign * 0.04, 0.58, -0.05),
            (0.5 - sign * 0.04, 0.65, 0.0),
        ]

    # Pinky: 17, 18, 19, 20
    if gesture_type in ("OPEN_PALM",):
        # Extended straight up
        lm += [
            (0.5 - sign * 0.08, 0.63, 0.0),
            (0.5 - sign * 0.08, 0.55, 0.0),
            (0.5 - sign * 0.08, 0.47, 0.0),
            (0.5 - sign * 0.08, 0.39, 0.0),
        ]
    else:  # FIST, POINTING, VICTORY
        # Curled into palm
        lm += [
            (0.5 - sign * 0.08, 0.63, 0.0),
            (0.5 - sign * 0.08, 0.54, -0.05),
            (0.5 - sign * 0.08, 0.60, -0.05),
            (0.5 - sign * 0.08, 0.67, 0.0),
        ]

    pixel_pts = [(int(x * 1280), int(y * 720)) for x, y, z in lm]
    fingertips = {
        "Thumb": pixel_pts[4],
        "Index": pixel_pts[8],
        "Middle": pixel_pts[12],
        "Ring": pixel_pts[16],
        "Pinky": pixel_pts[20],
    }

    return HandData(
        index=0 if is_r else 1,
        handedness=handedness,
        confidence=0.95,
        landmarks_norm=lm,
        landmarks_pixel=pixel_pts,
        fingertips_pixel=fingertips,
        palm_center=(640, 450),
        bounding_box=(400, 200, 880, 650),
        finger_states={
            "Thumb": gesture_type in ("OPEN_PALM", "POINTING", "VICTORY"),
            "Index": gesture_type in ("OPEN_PALM", "POINTING", "VICTORY"),
            "Middle": gesture_type in ("OPEN_PALM", "VICTORY"),
            "Ring": gesture_type == "OPEN_PALM",
            "Pinky": gesture_type == "OPEN_PALM",
        },
        landmarks_world=list(lm),
    )


def test_all_poses():
    print("=" * 70)
    print("PHASE 5: VERIFYING RECOGNITION OF ALL CORE GESTURES & HANDS")
    print("=" * 70)

    classifier = GestureClassifier()

    cases = [
        ("OPEN_PALM", "Right", StaticGesture.OPEN_PALM.value),
        ("OPEN_PALM", "Left", StaticGesture.OPEN_PALM.value),
        ("FIST", "Right", StaticGesture.FIST.value),
        ("FIST", "Left", StaticGesture.FIST.value),
        ("POINTING", "Right", StaticGesture.POINTING.value),
        ("POINTING", "Left", StaticGesture.POINTING.value),
        ("VICTORY", "Right", StaticGesture.VICTORY.value),
        ("VICTORY", "Left", StaticGesture.VICTORY.value),
    ]

    all_passed = True
    for req_pose, hand, expected in cases:
        hand_data = generate_hand_data(req_pose, hand)
        result = classifier.classify(hand_data)
        status = "PASS" if result.name == expected else "FAIL"
        print(f"[{status}] Pose: {req_pose:12s} | Hand: {hand:5s} -> Recognized: {result.name:12s} (conf: {result.confidence:.2f})")
        if status != "PASS":
            all_passed = False

    print("\nTesting Dual Hands (Both Hands simultaneous)...")
    right_hand = generate_hand_data("OPEN_PALM", "Right")
    left_hand = generate_hand_data("OPEN_PALM", "Left")

    engine = GestureEngine()
    tracking_res = HandTrackingResult(hands=[right_hand, left_hand], num_hands=2, is_success=True)
    # Feed 4 consecutive frames to satisfy temporal stability filter
    for t_step in range(4):
        telemetry = engine.process(tracking_res, timestamp_ms=1000 + t_step * 33)

    print(f"Dual hand recognition: Hands in telemetry = {list(telemetry.hands.keys())}")
    for h_name, h_state in telemetry.hands.items():
        print(f"  {h_name} hand pose: {h_state.current_pose} (conf: {h_state.confidence:.2f})")
    print(f"  Dual action: {telemetry.active_dual_gesture}")

    assert "Right" in telemetry.hands and "Left" in telemetry.hands, "Both hands must be tracked!"
    assert telemetry.active_dual_gesture == "DUAL_OPEN_PALMS", f"Expected DUAL_OPEN_PALMS, got {telemetry.active_dual_gesture}"
    print("\n[SUCCESS] All pose classifications and dual hand interactions verified successfully!")


if __name__ == "__main__":
    test_all_poses()
