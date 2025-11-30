import cv2
import numpy as np
import math

# ================= CẤU HÌNH INPUT =================
USE_IMAGE = True          # True = test 1 ảnh, False = video/webcam
USE_VIDEO_FILE = False      # nếu USE_IMAGE=False & USE_VIDEO_FILE=True => dùng video
IMAGE_PATH = "data\\test.jpg"
VIDEO_PATH = "IMG_1242.MOV"
# ==================================================

# Số frame dùng để calibrate ellipse
CALIB_FRAMES = 40          # tùy video, 30–60 đều ổn


def detect_edges_vong8(frame):
    """
    Dò vòng 8 bằng Top-hat + Otsu:
    - Làm nổi các đường sáng mỏng (vạch sơn) trên nền tối.
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tăng tương phản cục bộ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Top-hat: nhấn các chi tiết sáng mỏng
    kernel_size = 81  # bạn đang dùng ~88, để 81 cho số lẻ
    kernel_tophat = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel_tophat)

    # Otsu threshold
    _, mask = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ROI: chỉ giữ nửa dưới
    roi_mask = np.zeros_like(mask)
    roi_mask[int(h * 0.4):, :] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    # Mượt + CLOSE
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


def pick_vong8_candidates(contours, h, w):
    """
    Lọc contour khả thi là vòng 8.
    Ở phase CALIB, chúng ta chỉ cần ellipse "hợp lý", không cần tracking.
    """
    min_area = 8000
    max_area = 0.6 * w * h

    min_major_axis = 0.30 * w
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
        if cy < h * 0.55 or cy > h * 0.98:
            continue

        # trục lớn phải đủ dài (loại vũng nước)
        if MA < min_major_axis or MA > max_major_axis:
            continue

        # tỉ lệ trục: vòng 8 khá dẹt
        ratio = max(MA, ma) / (min(MA, ma) + 1e-6)
        if ratio < 1.1 or ratio > 4.0:
            continue

        candidates.append(((cx, cy), (MA, ma), angle, area, cnt))

    return candidates


def median_ellipse(ellipse_list):
    """Lấy median của nhiều ellipse để ra ellipse 'chuẩn'."""
    xs = [e[0][0] for e in ellipse_list]
    ys = [e[0][1] for e in ellipse_list]
    MAs = [e[1][0] for e in ellipse_list]
    mas = [e[1][1] for e in ellipse_list]
    angles = [e[2] for e in ellipse_list]

    cx = float(np.median(xs))
    cy = float(np.median(ys))
    MA = float(np.median(MAs))
    ma = float(np.median(mas))
    angle = float(np.median(angles))

    return ((cx, cy), (MA, ma), angle)


# ============ HÀM CHECK ĐIỂM TRONG / NGOÀI ELLIPSE ============

def point_inside_rotated_ellipse(x, y, ellipse):
    """
    Kiểm tra (x, y) nằm TRONG hay NGOÀI ellipse đã xoay.
    ellipse = ((cx, cy), (MA, ma), angle_deg)
    MA, ma là TRỤC DÀI (width), TRỤC NGẮN (height) - full length
    """
    (cx, cy), (MA, ma), angle_deg = ellipse

    # Bán trục
    a = MA / 2.0
    b = ma / 2.0

    # Đổi góc sang radian
    theta = np.deg2rad(angle_deg)

    # Đưa điểm về hệ trục trung tâm ellipse
    dx = x - cx
    dy = y - cy

    # Quay ngược lại theo góc ellipse để về hệ trục chuẩn
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xp = dx * cos_t + dy * sin_t
    yp = -dx * sin_t + dy * cos_t

    val = (xp ** 2) / (a ** 2 + 1e-6) + (yp ** 2) / (b ** 2 + 1e-6)

    # True nếu TRONG hoặc ngay trên biên
    return val <= 1.0, val


# ===============================================================


def process_frame(frame, frame_idx, calibrated_ellipse, ellipse_buffer,
                  prev_inside, hit_count):
    """
    Xử lý 1 frame:
    - Nếu chưa calibrate: phát hiện ellipse, lưu vào buffer.
    - Nếu đã calibrate: vẽ ellipse cố định.
    - DEMO PHASE 3: vẽ 1 'bánh xe ảo' chạy cắt qua ellipse và báo CHẠM VẠCH.
    """
    edges = detect_edges_vong8(frame)
    h, w = frame.shape[:2]
    overlay = frame.copy()

    mode_text = ""
    ellipse_to_draw = None

    # ------------------ PHASE 1: CALIB ------------------
    if calibrated_ellipse is None:
        mode_text = f"CALIBRATING {frame_idx+1}/{CALIB_FRAMES}"

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = pick_vong8_candidates(contours, h, w)

        if candidates:
            # chọn contour tốt nhất (diện tích lớn nhất)
            candidates.sort(key=lambda c: c[3], reverse=True)
            best = candidates[0]
            ellipse_raw = (best[0], best[1], best[2])

            ellipse_buffer.append(ellipse_raw)
            ellipse_to_draw = ellipse_raw

            # vẽ contour vàng cho debug
            cv2.drawContours(overlay, [best[4]], -1, (0, 255, 255), 1)

        # Sau khi đủ số frame, khóa ellipse
        if (frame_idx + 1) >= CALIB_FRAMES and ellipse_buffer:
            calibrated_ellipse = median_ellipse(ellipse_buffer)
            print(">> CALIB DONE:", calibrated_ellipse)
            mode_text = "RUN (ellipse locked)"
            ellipse_to_draw = calibrated_ellipse

    else:
        # ------------------ PHASE 2: RUN ------------------
        mode_text = "RUN (ellipse locked)"
        ellipse_to_draw = calibrated_ellipse

        # ====== DEMO PHASE 3: BÁNH XE ẢO CHẠY CẮT ELLIPSE ======
        (cx, cy), (MA, ma), angle = calibrated_ellipse
        a = MA / 2.0
        b = ma / 2.0

        # Bánh xe ảo: chạy từ trên xuống dưới, đi qua ellipse
        wheel_x = int(cx)  # cho chạy trên trục dọc qua tâm ellipse
        start_y = int(cy - b - 50)
        end_y   = int(cy + b + 50)
        if end_y <= start_y:
            end_y = start_y + 1

        # Dùng frame_idx để tạo chuyển động lặp
        span = end_y - start_y
        wheel_y = start_y + ((frame_idx - CALIB_FRAMES) % span)

        inside, val = point_inside_rotated_ellipse(
            wheel_x, wheel_y, calibrated_ellipse
        )

        # Vẽ bánh xe ảo
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(overlay, (wheel_x, wheel_y), 8, color, -1)

        # Text trạng thái
        state_text = "INSIDE" if inside else "OUTSIDE"
        cv2.putText(
            overlay,
            f"Wheel: {state_text}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Check CHẠM VẠCH: đổi trạng thái INSIDE -> OUTSIDE
        if prev_inside is True and inside is False:
            hit_count += 1
            print(f"CHAM VACH! hit_count={hit_count}")
            cv2.putText(
                overlay,
                "CHAM VACH!",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        prev_inside = inside

    # Vẽ ellipse nếu có
    if ellipse_to_draw is not None:
        cv2.ellipse(overlay, ellipse_to_draw, (0, 255, 0), 2)

    # Ghép khung hiển thị
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    show_top = np.hstack((frame, overlay, edges_bgr))
    show_bottom = np.hstack(
        (edges_bgr, np.zeros_like(edges_bgr), np.zeros_like(edges_bgr))
    )
    show = np.vstack((show_top, show_bottom))

    # overlay text mode
    cv2.putText(
        show,
        mode_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        show,
        f"Hits: {hit_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    show = cv2.resize(show, None, fx=0.35, fy=0.35)
    return show, calibrated_ellipse, ellipse_buffer, prev_inside, hit_count


def main():
    calibrated_ellipse = None
    ellipse_buffer = []
    frame_idx = 0

    # trạng thái để demo CHẠM VẠCH
    prev_inside = None
    hit_count = 0

    if USE_IMAGE:
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print("Không đọc được ảnh:", IMAGE_PATH)
            return

        # 1) Tự detect ellipse 1 lần trên ảnh này
        edges = detect_edges_vong8(frame)
        h, w = frame.shape[:2]
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = pick_vong8_candidates(contours, h, w)

        if not candidates:
            print("Không tìm được contour vòng 8 trong ảnh.")
            return

        candidates.sort(key=lambda c: c[3], reverse=True)
        best = candidates[0]
        calibrated_ellipse = (best[0], best[1], best[2])
        print(">> CALIB ELLIPSE FROM IMAGE:", calibrated_ellipse)

        # 2) Demo: cho bánh xe ảo chạy qua ellipse trên chính ảnh này
        prev_inside = None
        hit_count = 0
        frame_idx = CALIB_FRAMES  # cho vào thẳng phase RUN

        while True:
            show, calibrated_ellipse, _, prev_inside, hit_count = process_frame(
                frame, frame_idx, calibrated_ellipse, [], prev_inside, hit_count
            )
            frame_idx += 1

            cv2.imshow("Vong 8 - Image demo (wheel test)", show)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break

        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
