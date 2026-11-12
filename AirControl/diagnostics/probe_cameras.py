import cv2
import time

print("Testing Camera Index 0 continuous capture and auto-exposure...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for i in range(40):
    ret, frame = cap.read()
    if ret and frame is not None:
        mean_val = float(frame.mean())
        if i % 5 == 0 or i == 39:
            print(f"Frame {i:2d}: mean={mean_val:.2f}, min={frame.min()}, max={frame.max()}")
    time.sleep(0.03)

if ret and frame is not None:
    cv2.imwrite("diagnostics/frame_sample.jpg", frame)
    print("Saved diagnostics/frame_sample.jpg")

cap.release()
