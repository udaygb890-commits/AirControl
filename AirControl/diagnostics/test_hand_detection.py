"""
AirControl — Independent MediaPipe Hand Detection Diagnostic Script
Used to isolate and verify MediaPipe Hand Landmarker detection directly against webcam frames.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Locate assets
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "assets" / "hand_landmarker.task"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """Draw hand landmarks and skeleton connections on the image."""
    annotated = rgb_image.copy()
    h, w = annotated.shape[:2]

    if not detection_result or not detection_result.hand_landmarks:
        return annotated

    for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
        # Determine handedness label if available
        handedness_label = "Hand"
        score = 0.0
        if detection_result.handedness and hand_idx < len(detection_result.handedness):
            cat = detection_result.handedness[hand_idx][0]
            handedness_label = cat.category_name or cat.display_name or "Hand"
            score = cat.score

        # Convert landmarks to pixel coordinates
        pixel_points = []
        for lm in landmarks:
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            pixel_points.append((px, py))

        # Draw bone connections
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            cv2.line(annotated, pixel_points[p1_idx], pixel_points[p2_idx], (0, 255, 255), 2)

        # Draw joint landmarks
        for px, py in pixel_points:
            cv2.circle(annotated, (px, py), 4, (0, 128, 255), -1)
            cv2.circle(annotated, (px, py), 2, (255, 255, 255), -1)

        # Draw hand label at wrist
        wrist_x, wrist_y = pixel_points[0]
        text = f"{handedness_label} ({score:.2f})"
        cv2.putText(annotated, text, (wrist_x - 20, max(20, wrist_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated


def run_diagnostics(duration_sec: float = 0.0, show_window: bool = True):
    print("=" * 70)
    print("AIRCONTROL — INDEPENDENT MEDIAPIPE HAND DETECTION TEST")
    print("=" * 70)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"MediaPipe:    {mp.__version__}")
    print(f"OpenCV:       {cv2.__version__}")
    print(f"Model path:   {MODEL_PATH}")
    print(f"Model exists: {MODEL_PATH.exists()} (size: {MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0} bytes)")
    print(f"Camera index: {CAMERA_INDEX} ({FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS}fps)")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found at {MODEL_PATH}")
        return False

    # 1. Initialize MediaPipe HandLandmarker in VIDEO mode
    print("\n[PHASE 1] Initializing MediaPipe HandLandmarker Tasks API (RunningMode.VIDEO)...")
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        landmarker = vision.HandLandmarker.create_from_options(options)
        print("[SUCCESS] HandLandmarker initialized successfully.")
    except Exception as e:
        print(f"[FATAL] Failed to create HandLandmarker: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Open Camera
    print(f"\n[PHASE 2] Opening VideoCapture on index {CAMERA_INDEX}...")
    cap = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        test_cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if test_cap.isOpened():
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                cap = test_cap
                backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else ("MSMF" if backend == cv2.CAP_MSMF else "ANY")
                print(f"[SUCCESS] Camera opened with backend {backend_name}.")
                break
            test_cap.release()

    if cap is None:
        print(f"[ERROR] Could not open camera {CAMERA_INDEX} on any backend.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Sensor stream active: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    # 3. Capture & Detection Loop
    print("\n[PHASE 3] Starting detection loop. Press 'q' in the window or Ctrl+C in terminal to exit.")
    print("Printing statistics every second...\n")

    start_time = time.time()
    last_print_time = start_time
    last_timestamp_ms = -1
    frame_count = 0
    total_detected_frames = 0

    window_title = "AirControl — MediaPipe Standalone Diagnostic Test"

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None or bgr_frame.size == 0:
                print("[WARNING] Frame capture returned empty/failed.")
                time.sleep(0.03)
                continue

            frame_count += 1
            now = time.time()

            # Monotonically increasing timestamp in milliseconds
            timestamp_ms = int(now * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            # Mirror frame horizontally (standard selfie camera view)
            bgr_frame = cv2.flip(bgr_frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # Construct MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect
            t0 = time.perf_counter()
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            num_2d_hands = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
            raw_world = getattr(detection_result, "hand_world_landmarks", None)
            num_world_sets = len(raw_world) if raw_world else 0
            num_handedness = len(detection_result.handedness) if detection_result.handedness else 0

            if num_2d_hands > 0:
                total_detected_frames += 1

            # Print every 1.0 second
            if now - last_print_time >= 1.0:
                fps = frame_count / (now - start_time)
                hand_labels = []
                if detection_result.handedness:
                    for h_list in detection_result.handedness:
                        if h_list:
                            cat = h_list[0]
                            # Mirror flipped:
                            name = "Right" if cat.category_name == "Left" else "Left"
                            hand_labels.append(f"{name}:{cat.score:.2f}")

                labels_str = f"[{', '.join(hand_labels)}]" if hand_labels else "[]"
                print(
                    f"[{time.strftime('%H:%M:%S')}] FPS: {fps:4.1f} | "
                    f"Inference: {infer_ms:4.1f}ms | "
                    f"2D Hands: {num_2d_hands}/2 | "
                    f"World Landmark Sets: {num_world_sets} | "
                    f"Handedness: {num_handedness} {labels_str}"
                )
                last_print_time = now

            # Draw preview if window is enabled
            if show_window:
                annotated_rgb = draw_landmarks_on_image(rgb_frame, detection_result)
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                hud_text = f"2D Hands: {num_2d_hands} | World Sets: {num_world_sets} | Infer: {infer_ms:.1f}ms"
                hud_col = (0, 255, 0) if num_2d_hands > 0 else (0, 165, 255)
                cv2.putText(annotated_bgr, hud_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_col, 2)

                try:
                    cv2.imshow(window_title, annotated_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("User terminated diagnostic preview.")
                        break
                except cv2.error:
                    show_window = False

            if duration_sec > 0 and (now - start_time) >= duration_sec:
                print(f"\nCompleted diagnostic duration of {duration_sec}s.")
                break

    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user.")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY:")
    print(f"Total time:       {elapsed:.1f}s")
    print(f"Total frames:     {frame_count}")
    print(f"Frames with hand: {total_detected_frames} ({total_detected_frames / max(1, frame_count) * 100:.1f}%)")
    print("=" * 70)
    return total_detected_frames > 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AirControl MediaPipe Hand Detection Diagnostic")
    parser.add_argument("--duration", type=float, default=0.0, help="Test duration in seconds (0 = infinite)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless terminal mode without OpenCV GUI window")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    CAMERA_INDEX = args.camera
    run_diagnostics(duration_sec=args.duration, show_window=not args.no_gui)
