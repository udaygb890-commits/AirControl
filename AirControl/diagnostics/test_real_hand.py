import cv2
import numpy as np
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from vision.hand_tracker import HandTracker

# Let's create a synthetic realistic image with a hand or load an image
# First let's check if we have any test images or if we can download a hand test image
import urllib.request

test_img_path = Path("diagnostics/test_hand.jpg")
if not test_img_path.exists():
    # Download a standard public domain hand image
    url = "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu/res/drawable/hand.jpg"
    try:
        print("Downloading official MediaPipe test hand image...")
        req = urllib.request.Request(
            "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as resp, open(str(test_img_path), "wb") as f:
            f.write(resp.read())
        print("Downloaded hand image successfully.")
    except Exception as e:
        print("Download error:", e)

if test_img_path.exists():
    img = cv2.imread(str(test_img_path))
    print(f"Loaded image: shape={img.shape}")
    
    # 1. Test directly with HandLandmarker
    model_path = Path("assets/hand_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    res1 = landmarker.detect_for_video(mp_image, 100)
    print("Standalone Landmarker Hands:", len(res1.hand_landmarks) if res1.hand_landmarks else 0)
    if res1.hand_landmarks:
        print("  Confidence:", res1.handedness[0][0].score if res1.handedness else "N/A")
        print("  World landmarks present:", bool(res1.hand_world_landmarks))
        
    # 2. Test with HandTracker
    tracker = HandTracker()
    tracker.initialize()
    res2 = tracker.process_frame(img, 200)
    print("HandTracker Hands:", res2.num_hands)
    if res2.hands:
        print("  Handedness:", res2.hands[0].handedness)
        print("  Confidence:", res2.hands[0].confidence)
        print("  3D Model Valid:", bool(res2.hands[0].hand_model_3d and res2.hands[0].hand_model_3d.is_valid))
