import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "IMG_1242.MOV"
WINDOW_NAME = "Level3 - Bike + Feet + Wheels"

# ----- Model paths -----
BIKE_MODEL_PATH = "yolov8n.pt"          # COCO
POSE_MODEL_PATH = "yolov8n-pose.pt"     # COCO pose
WHEEL_MODEL_PATH = "wheel_best.pt"      # custom 2-class model: 0=rear,1=front (nếu có)

MAX_TRAIL_LEN = 150         # giới hạn độ dài trail
MISSING_RESET_FRAMES = 15   # nếu mất xe > 15 frame => reset trail


def clamp_trail(trail, max_len=MAX_TRAIL_LEN):
    if len(trail) > max_len:
        trail[:] = trail[-max_len:]


def find_best_bike(result, min_conf=0.4):
    if result.boxes is None:
        return None
    best = None
    best_conf = 0.0

    for b in result.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        if cls == 3 and conf >= min_conf:   # 3 = motorbike
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if conf > best_conf and (x2-x1) > 10 and (y2-y1) > 10:
                best_conf = conf
                best = (x1, y1, x2-x1, y2-y1)
    return best


def find_person_for_bike(pose, bike_box):
    if bike_box is None or pose.boxes is None:
        return None

    bx, by, bw, bh = bike_box
    bcx, bcy = bx + bw/2, by + bh/2

    best_idx = None
    best_dist = 1e9
    for i, box in enumerate(pose.boxes):
        x1, y1, x2, y2 = box.xyxy[0]
        px, py = float((x1+x2)/2), float((y1+y2)/2)
        d = (px - bcx)**2 + (py - bcy)**2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def detect_wheels_yolo(frame, wheel_model, bike_box, min_conf=0.4):
    """
    Dùng model YOLO riêng cho bánh xe.
    Giả sử:
      class 0 = rear_wheel
      class 1 = front_wheel
    """
    if wheel_model is None or bike_box is None:
        return None, None

    bx, by, bw, bh = bike_box

    # crop nhẹ để giảm noise
    pad_y = int(0.25 * bh)
    y1 = max(by, by + bh//2 - pad_y)
    y2 = min(by + bh, by + bh + pad_y)
    x1 = bx
    x2 = bx + bw

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    res = wheel_model(roi, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, None

    rear = None
    front = None

    for b in res.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        if conf < min_conf:
            continue
        wx1, wy1, wx2, wy2 = b.xyxy[0]
        wx = int((wx1 + wx2) / 2)
        wy = int((wy1 + wy2) / 2)

        # map về toạ độ frame gốc
        gx = x1 + wx
        gy = y1 + wy

        if cls == 0:   # rear
            rear = (gx, gy)
        elif cls == 1: # front
            front = (gx, gy)

    return rear, front


def detect_wheels_hough(frame, bike_box):
    """
    Fallback: dùng HoughCircle tìm 1–2 bánh trong vùng gầm xe.
    """
    if bike_box is None:
        return None, None

    bx, by, bw, bh = bike_box

    # ROI gầm xe (1/2 dưới khung)
    y1 = by + int(0.45 * bh)
    y2 = by + bh
    x1 = bx
    x2 = bx + bw

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    minR = int(0.10 * bh)
    maxR = int(0.40 * bh)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=bw / 3,
        param1=120,
        param2=25,
        minRadius=minR,
        maxRadius=maxR,
    )

    if circles is None:
        return None, None

    circles = np.round(circles[0, :]).astype(int)
    # sort theo x (trái = rear, phải = front)
    circles = sorted(circles, key=lambda c: c[0])

    rear = None
    front = None

    if len(circles) >= 1:
        cx, cy, r = circles[0]
        rear = (x1 + cx, y1 + cy)
    if len(circles) >= 2:
        cx, cy, r = circles[-1]
        front = (x1 + cx, y1 + cy)

    return rear, front


def estimate_wheels_from_box(bike_box):
    bx, by, bw, bh = bike_box
    wy = by + bh - int(0.05 * bh)
    rear = (bx + int(0.25*bw), wy)
    front = (bx + int(0.75*bw), wy)
    return rear, front


def main():
    if not os.path.exists(VIDEO_PATH):
        print("Video not found:", VIDEO_PATH)
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ----- MODELS -----
    bike_model = YOLO(BIKE_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    wheel_model = None
    if os.path.exists(WHEEL_MODEL_PATH):
        wheel_model = YOLO(WHEEL_MODEL_PATH)
        print("Wheel model loaded:", WHEEL_MODEL_PATH)
    else:
        print("⚠ Không tìm thấy wheel_best.pt, chỉ dùng Hough + ước lượng box cho bánh.")

    # ----- STATE -----
    paused = False
    seek_requested = False   # đánh dấu vừa kéo seekbar

    show_trail = True
    bike_trail = []
    rear_trail = []
    front_trail = []
    leftfoot_trail = []
    rightfoot_trail = []
    missing_count = 0

    # ----- SEEK BAR -----
    def on_seek(pos):
        nonlocal paused, seek_requested, cap
        paused = True                 # kéo là tự pause
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        seek_requested = True         # báo vòng while đọc 1 frame mới

    cv2.createTrackbar("Seek", WINDOW_NAME, 0, total_frames - 1, on_seek)

    frame = None  # để tránh dùng trước khi gán

    while True:
        # đọc frame mới khi:
        #  - đang chạy bình thường (not paused)
        #  - HOẶC vừa kéo seekbar (seek_requested)
        if not paused or seek_requested:
            ret, frame = cap.read()
            if not ret:
                break
            seek_requested = False

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("Seek", WINDOW_NAME, frame_idx)

        if frame is None:
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        # ===== Detect bike =====
        bike_res = bike_model(frame, verbose=False)[0]
        bike_box = find_best_bike(bike_res)

        if bike_box is None:
            missing_count += 1
        else:
            missing_count = 0

        # nếu mất xe nhiều frame -> reset trail cho đỡ rối
        if missing_count > MISSING_RESET_FRAMES:
            bike_trail.clear()
            rear_trail.clear()
            front_trail.clear()
            leftfoot_trail.clear()
            rightfoot_trail.clear()

        # ===== Detect rider pose =====
        pose_res = pose_model(frame, verbose=False)[0]
        rider_idx = find_person_for_bike(pose_res, bike_box)

        # ===== Draw bike & wheels =====
        rear = None
        front = None

        if bike_box is not None:
            bx, by, bw, bh = bike_box
            bcx, bcy = bx + bw // 2, by + bh // 2

            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(display, (bcx, bcy), 4, (0, 255, 0), -1)

            # 1) YOLO bánh (nếu có)
            rear, front = detect_wheels_yolo(frame, wheel_model, bike_box)
            # 2) Hough fallback
            if rear is None or front is None:
                r2, f2 = detect_wheels_hough(frame, bike_box)
                if rear is None:
                    rear = r2
                if front is None:
                    front = f2
            # 3) ước lượng cuối
            if rear is None or front is None:
                er, ef = estimate_wheels_from_box(bike_box)
                if rear is None:
                    rear = er
                if front is None:
                    front = ef

            if rear is not None:
                cv2.circle(display, rear, 5, (0, 255, 255), -1)
            if front is not None:
                cv2.circle(display, front, 5, (0, 255, 255), -1)

            if show_trail:
                bike_trail.append((bcx, bcy))
                clamp_trail(bike_trail)
                if rear is not None:
                    rear_trail.append(rear)
                    clamp_trail(rear_trail)
                if front is not None:
                    front_trail.append(front)
                    clamp_trail(front_trail)

                for i in range(1, len(bike_trail)):
                    cv2.line(display, bike_trail[i-1], bike_trail[i], (0, 255, 0), 2)
                for i in range(1, len(rear_trail)):
                    cv2.line(display, rear_trail[i-1], rear_trail[i], (0, 255, 255), 2)
                for i in range(1, len(front_trail)):
                    cv2.line(display, front_trail[i-1], front_trail[i], (0, 255, 255), 2)

        # ===== Draw feet =====
        if rider_idx is not None and pose_res.keypoints is not None:
            kps = pose_res.keypoints.xy[rider_idx]
            lx, ly = map(int, kps[15])  # left_ankle
            rx, ry = map(int, kps[16])  # right_ankle

            cv2.circle(display, (lx, ly), 6, (0, 0, 255), -1)
            cv2.circle(display, (rx, ry), 6, (255, 0, 0), -1)

            if show_trail:
                leftfoot_trail.append((lx, ly))
                rightfoot_trail.append((rx, ry))
                clamp_trail(leftfoot_trail)
                clamp_trail(rightfoot_trail)

                for i in range(1, len(leftfoot_trail)):
                    cv2.line(display, leftfoot_trail[i-1], leftfoot_trail[i], (0, 0, 255), 2)
                for i in range(1, len(rightfoot_trail)):
                    cv2.line(display, rightfoot_trail[i-1], rightfoot_trail[i], (255, 0, 0), 2)

        # ===== UI text =====
        cv2.putText(
            display,
            f"Frame {frame_idx}/{total_frames}  |  trail={'ON' if show_trail else 'OFF'}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('t'):
            show_trail = not show_trail
        if key == ord('r'):
            bike_trail.clear()
            rear_trail.clear()
            front_trail.clear()
            leftfoot_trail.clear()
            rightfoot_trail.clear()

        # mũi tên → / ← (nếu hệ thống bạn bắt được) vẫn giữ
        if key == 83:   # →
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, frame_idx + 50))
        if key == 81:   # ←
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 50))

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
