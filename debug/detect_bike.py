import cv2
import sys
import os

# ===============================
# CHỌN CHẾ ĐỘ INPUT
# ===============================
# "webcam"  → dùng webcam
# "video"   → dùng video file
# "image"   → dùng 1 ảnh tĩnh
INPUT_MODE = "video"

VIDEO_PATH = "xe.mp4"    # đổi đường dẫn video
IMAGE_PATH = "xe.jpg"    # đổi đường dẫn ảnh


# ===============================
# TẠO TRACKER
# ===============================
def create_tracker():
    # CSRT = chính xác, dùng tốt cho xe
    if hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()
    else:
        return cv2.TrackerCSRT_create()


# ===============================
# LẤY FRAME ĐẦU
# ===============================
def get_first_frame():
    global cap

    if INPUT_MODE == "webcam":
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        return ok, frame

    elif INPUT_MODE == "video":
        if not os.path.exists(VIDEO_PATH):
            print("Không tìm thấy video:", VIDEO_PATH)
            sys.exit()
        cap = cv2.VideoCapture(VIDEO_PATH)
        ok, frame = cap.read()
        return ok, frame

    elif INPUT_MODE == "image":
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print("Không đọc được ảnh:", IMAGE_PATH)
            sys.exit()
        return True, frame

    else:
        print("INPUT_MODE không hợp lệ")
        sys.exit()


# ===============================
# CHẠY CHƯƠNG TRÌNH
# ===============================
ok, frame = get_first_frame()
if not ok:
    print("Không lấy được frame đầu.")
    sys.exit()

# CHỌN VÙNG XE
roi = cv2.selectROI("Chọn xe để track", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Chọn xe để track")

x, y, w, h = roi
if w == 0 or h == 0:
    print("Không chọn vùng, thoát.")
    sys.exit()

tracker = create_tracker()
tracker.init(frame, roi)

trail = []
frame_id = 0

# Nếu là ảnh → chỉ chạy 1 frame
if INPUT_MODE == "image":
    cx = x + w // 2
    cy = y + h // 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.imshow("Debug Track Xe (Ảnh)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()


# ===============================
# LOOP CHO VIDEO / WEBCAM
# ===============================
while True:
    ok, frame = cap.read()
    if not ok:
        print("Hết video hoặc mất frame.")
        break

    frame_id += 1
    ok, box = tracker.update(frame)

    if ok:
        x, y, w, h = [int(v) for v in box]
        cx = x + w // 2
        cy = y + h // 2

        # Vẽ khung
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Trail
        trail.append((cx, cy))
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (255, 0, 0), 2)

        cv2.putText(frame, "Tracking OK", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"[Frame {frame_id}] center=({cx},{cy}) box=({x},{y},{w},{h})")

    else:
        cv2.putText(frame, "Tracking LOST", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print(f"[Frame {frame_id}] LOST")

    cv2.imshow("Debug Track Xe", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
