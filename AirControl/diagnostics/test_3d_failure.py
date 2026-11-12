import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from vision.hand_tracker import HandTracker

test_img = cv2.imread("diagnostics/test_hand.jpg")
tracker = HandTracker()
tracker.initialize()

# Normal detection
res = tracker.process_frame(test_img, 100)
print("Normal detection - Hands:", res.num_hands, "Success:", res.is_success)
assert res.num_hands == 2
assert res.hands[0].hand_model_3d is not None

# Now test when 3D processor raises an exception
tracker._processor_right.update = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Simulated 3D kinematic crash"))

res_fault = tracker.process_frame(test_img, 200)
print("Fault injection (3D failure) - Hands:", res_fault.num_hands, "Success:", res_fault.is_success)
print("  Hand 0 2D Landmarks:", len(res_fault.hands[0].landmarks_norm))
print("  Hand 0 3D Model:", res_fault.hands[0].hand_model_3d)
assert res_fault.num_hands == 2, "2D detection must remain valid when 3D fails!"
assert res_fault.hands[0].hand_model_3d is None, "3D model must be None on failure"
print("\n[VERIFICATION PASSED] 2D hand detection is authoritative and completely resilient against 3D failures!")

