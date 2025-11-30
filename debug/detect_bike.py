import cv2
import os
import sys

VIDEO_PATH = "IMG_1242.MOV"
WINDOW_NAME = "Auto Track Xe"

def create_tracker():
    if hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()
    return cv2.TrackerCSRT_create()

def main():
    if not os.path.exists(VIDEO_PATH):
        print("Không tìm thấy video:", VIDEO_PATH)
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Không mở được video:", VIDEO_PATH)
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # bộ trừ nền (background subtractor)
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )

    tracker = None
    has_tracker = False
    trail = []
    frame_idx = 0

    MIN_AREA = 5000  # tùy video, chỉnh cho hợp (pixel)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video.")
            break

        frame_idx += 1
        display = frame.copy()

        # ============ NẾU CHƯA CÓ TRACKER → TỰ TÌM XE ============
        if not has_tracker:
            fg = backsub.apply(frame)           # lấy vùng chuyển động
            fg = cv2.medianBlur(fg, 5)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
                                  iterations=2)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            big_candidate = None
            big_area = 0

            for c in contours:
                area = cv2.contourArea(c)
                if area < MIN_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # có thể thêm điều kiện: nằm trong vùng đường thi, v.v.
                if area > big_area:
                    big_area = area
                    big_candidate = (x, y, w, h)

            if big_candidate is not None:
                x, y, w, h = big_candidate
                # init tracker lần đầu tiên
                tracker = create_tracker()
                tracker.init(frame, (x, y, w, h))
                has_tracker = True
                trail = []
                print(f">> AUTO INIT TRACKER tại frame {frame_idx}, box={big_candidate}")
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
                status = f"Auto-init tracker | Frame {frame_idx}"
                color = (0, 255, 255)
            else:
                status = f"Đang chờ xe xuất hiện... Frame {frame_idx}"
                color = (0, 255, 255)

        # ============ NẾU ĐÃ CÓ TRACKER → TRACK BÌNH THƯỜNG ============
        else:
            ok, box = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in box]
                cx = x + w // 2
                cy = y + h // 2

                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

                trail.append((cx, cy))
                for i in range(1, len(trail)):
                    cv2.line(display, trail[i-1], trail[i], (255, 0, 0), 2)

                status = f"Tracking OK | Frame {frame_idx}"
                color = (0, 255, 0)
            else:
                status = f"Tracking LOST | Frame {frame_idx}"
                color = (0, 0, 255)

        cv2.putText(display, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
