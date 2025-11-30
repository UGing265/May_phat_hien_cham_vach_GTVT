import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "IMG_1242.MOV"
WINDOW_NAME = "YOLO Bike Debug"

# COCO pretrained model
MODEL_PATH = "yolov8n.pt"  # để cùng folder hoặc để ultralytics tự tải


def main():
    if not os.path.exists(VIDEO_PATH):
        print("Video not found:", VIDEO_PATH)
        sys.exit(1)

    # load YOLO model
    model = YOLO(MODEL_PATH)  # lần đầu sẽ tự tải nếu chưa có

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH)
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    trail = []
    delay = 30  # ms (≈ 33 FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_idx += 1
        display = frame.copy()

        # ----- YOLO detect every frame -----
        # classes=[3] => motorbike in COCO
        results = model(frame, verbose=False, conf=0.5, classes=[3])

        bike_box = None
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                # chọn box có confidence cao nhất
                best_conf = 0.0
                best = None
                for b in r.boxes:
                    conf = float(b.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best = b
                if best is not None:
                    x1, y1, x2, y2 = best.xyxy[0]
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    w = x2 - x1
                    h = y2 - y1
                    bike_box = (x1, y1, w, h)

        if bike_box is not None:
            x, y, w, h = bike_box
            cx = x + w // 2
            cy = y + h // 2

            # vẽ box + tâm
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

            # vẽ trail
            trail.append((cx, cy))
            for i in range(1, len(trail)):
                cv2.line(display, trail[i - 1], trail[i], (255, 0, 0), 2)

            text = f"Bike detected | Frame {frame_idx}"
            color = (0, 255, 0)
        else:
            text = f"No bike | Frame {frame_idx}"
            color = (0, 255, 255)

        cv2.putText(
            display,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(delay) & 0xFF

        if key in (27, ord("q")):
            break

        # pause / play
        if key == ord(" "):
            delay = 0 if delay != 0 else 30

        # speed control
        if key in (ord("+"), ord("=")):
            delay = max(1, delay - 5)
        if key in (ord("-"), ord("_")):
            delay += 5

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
