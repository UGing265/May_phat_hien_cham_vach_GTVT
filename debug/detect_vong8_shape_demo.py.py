import cv2
import numpy as np
import math

# ================= CẤU HÌNH INPUT =================
# Chọn 1 trong 3:
USE_IMAGE = True          # True = xử lý 1 ảnh
USE_VIDEO_FILE = False      # True = xử lý file video
# Nếu cả 2 đều False -> dùng webcam (camera 0)

IMAGE_PATH = "test.jpg"    # đường dẫn ảnh cần phân tích
VIDEO_PATH = "test.mp4"    # đường dẫn video cần phân tích
# ===================================================


def detect_edges_vong8(frame):
    """
    Dò vòng 8 bằng Top-hat + Otsu:
    - Làm nổi các đường sáng mỏng (vạch sơn) trên nền tối.
    - Không phụ thuộc nhiều vào màu trắng HSV nữa.
    """
    h, w = frame.shape[:2]

    # 1. BGR -> GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Tăng tương phản cục bộ (cho vạch rõ hơn nền)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # 3. Top-hat: nhấn mạnh các vùng sáng nhỏ trên nền tối
    # kernel càng to -> càng bắt các "vòng tròn lớn"
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel_tophat)

    # 4. Tự động chọn ngưỡng bằng Otsu
    _, mask = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5. ROI: chỉ giữ nửa dưới khung hình (nếu cần có thể hạ xuống 0.3)
    roi_mask = np.zeros_like(mask)
    roi_mask[int(h * 0.4):, :] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    # 6. Làm mượt + CLOSE để nối vòng
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask



def pick_vong8_candidates(contours, h, w):
    """Lọc contour khả thi là vòng 8."""
    min_area = 8000              # tăng lên, loại bớt vùng nhỏ (vũng nước)
    max_area = 0.6 * w * h

    min_major_axis = 0.30 * w    # MA (bán kính lớn) phải đủ dài
    max_major_axis = 0.95 * w

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if len(cnt) < 5:
            continue

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)

        # tâm ellipse phải ở nửa dưới
        if cy < h * 0.55 or cy > h * 0.95:
            continue

        # trục lớn phải đủ dài (loại vũng nước nhỏ xíu)
        if MA < min_major_axis or MA > max_major_axis:
            continue

        # tỉ lệ trục: vòng 8 khá dẹt, không tròn xoe
        ratio = max(MA, ma) / (min(MA, ma) + 1e-6)
        if ratio < 1.2 or ratio > 4.0:
            continue

        candidates.append(((cx, cy), (MA, ma), angle, area, cnt))

    return candidates



def smooth_ellipse(new_ellipse, prev_ellipse, alpha=0.2):
    """Làm mượt ellipse mới với ellipse cũ."""
    if prev_ellipse is None:
        return new_ellipse

    (cx, cy), (MA, ma), angle = new_ellipse
    (pcx, pcy), (pMA, pma), pangle = prev_ellipse

    sx = alpha * cx + (1 - alpha) * pcx
    sy = alpha * cy + (1 - alpha) * pcy
    sMA = alpha * MA + (1 - alpha) * pMA
    sma = alpha * ma + (1 - alpha) * pma
    sangle = alpha * angle + (1 - alpha) * pangle

    return ((sx, sy), (sMA, sma), sangle)


def ellipse_area(e):
    (cx, cy), (MA, ma), angle = e
    return math.pi * (MA / 2.0) * (ma / 2.0)


def process_frame(frame, prev_ellipse, lost_frames, max_keep_lost=5):
    """Xử lý 1 frame: detect vòng 8 + tracking + debug UI."""
    edges = detect_edges_vong8(frame)  # giờ edges là mask trắng

    # 1) Morphology CLOSE
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) Contour
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = frame.shape[:2]
    overlay = frame.copy()

    candidates = pick_vong8_candidates(contours, h, w)

    ellipse_to_draw = None

    if candidates:
        # có contour ứng viên
        if prev_ellipse is not None:
            pcx, pcy = prev_ellipse[0]
            # ưu tiên contour gần ellipse cũ
            candidates.sort(
                key=lambda c: (c[0][0] - pcx) ** 2 + (c[0][1] - pcy) ** 2
            )
        else:
            # chưa có ellipse cũ → chọn contour lớn nhất
            candidates.sort(key=lambda c: c[3], reverse=True)

        best = candidates[0]
        ellipse_raw = (best[0], best[1], best[2])

        accept = True
        if prev_ellipse is not None:
            # 1) check tâm nhảy quá xa không
            cx, cy = ellipse_raw[0]
            pcx, pcy = prev_ellipse[0]
            dist = math.hypot(cx - pcx, cy - pcy)
            max_move = 0.15 * math.hypot(w, h)  # cho phép nhảy ~15% đường chéo
            if dist > max_move:
                accept = False

            # 2) check diện tích khác quá nhiều không
            area_new = ellipse_area(ellipse_raw)
            area_old = ellipse_area(prev_ellipse)
            if not (0.5 * area_old <= area_new <= 1.5 * area_old):
                accept = False

        if accept:
            ellipse_to_draw = smooth_ellipse(ellipse_raw, prev_ellipse, alpha=0.2)
            prev_ellipse = ellipse_to_draw
            lost_frames = 0
            # vẽ contour debug
            cv2.drawContours(overlay, [best[4]], -1, (0, 255, 255), 1)
        else:
            # contour này “nhảy” quá đà → bỏ, chỉ dùng ellipse cũ
            if lost_frames < max_keep_lost:
                ellipse_to_draw = prev_ellipse
                lost_frames += 1
            else:
                prev_ellipse = None
                ellipse_to_draw = None
    else:
        # không có contour phù hợp
        if prev_ellipse is not None and lost_frames < max_keep_lost:
            ellipse_to_draw = prev_ellipse
            lost_frames += 1
        else:
            prev_ellipse = None
            ellipse_to_draw = None

    # Vẽ ellipse màu xanh
    if ellipse_to_draw is not None:
        cv2.ellipse(overlay, ellipse_to_draw, (0, 255, 0), 2)

    # 4) Ghép khung hiển thị
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    closed_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

    show_top = np.hstack((frame, overlay, edges_bgr))
    show_bottom = np.hstack(
        (closed_bgr, np.zeros_like(closed_bgr), np.zeros_like(closed_bgr))
    )
    show = np.vstack((show_top, show_bottom))

    show = cv2.resize(show, None, fx=0.35, fy=0.35)

    return show, prev_ellipse, lost_frames


def main():
    prev_ellipse = None
    lost_frames = 0

    # ====== CHỌN NGUỒN INPUT ======
    if USE_IMAGE:
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print("Không đọc được ảnh:", IMAGE_PATH)
            return

        show, prev_ellipse, lost_frames = process_frame(
            frame, prev_ellipse, lost_frames
        )
        cv2.imshow("Vong 8 - Image analysis", show)
        print("Nhấn phím bất kỳ để thoát...")
        cv2.waitKey(0)

    elif USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("Không mở được video:", VIDEO_PATH)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            show, prev_ellipse, lost_frames = process_frame(
                frame, prev_ellipse, lost_frames
            )
            cv2.imshow("Vong 8 - Video analysis", show)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC hoặc q
                break

        cap.release()

    else:
        # Webcam (giống code cũ)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không mở được camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            show, prev_ellipse, lost_frames = process_frame(
                frame, prev_ellipse, lost_frames
            )
            cv2.imshow("Vong 8 - Webcam", show)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
