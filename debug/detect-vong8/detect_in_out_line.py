import cv2
import numpy as np
import math

# ================= CẤU HÌNH INPUT =================
USE_IMAGE = False          # True = test 1 ảnh, False = video/webcam
USE_VIDEO_FILE = True      # nếu USE_IMAGE=False & USE_VIDEO_FILE=True => dùng video
IMAGE_PATH = "data/test.jpg"
VIDEO_PATH = "IMG_1242.MOV"
# ==================================================

# Số frame dùng để calibrate ellipse
CALIB_FRAMES = 40          # tùy video, 30–60 đều ổn

# NỬA BỀ RỘNG LÀN ĐƯỜNG (pixel) -> dùng để tạo vạch TRONG / NGOÀI
LANE_HALF_WIDTH = 40


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


def make_out_in_ellipses(ellipse_mid, lane_half_width):
    """
    Từ ellipse giữa (mid) tạo ra ellipse OUTER/INNER.
    ellipse_mid = ((cx, cy), (MA, ma), angle)
    lane_half_width: nửa bề rộng làn (pixel)
    """
    (cx, cy), (MA, ma), angle = ellipse_mid

    # MA, ma là full length => cộng/trừ 2 * lane_half_width
    MA_outer = MA + 2 * lane_half_width
    ma_outer = ma + 2 * lane_half_width
    MA_inner = max(MA - 2 * lane_half_width, 10)
    ma_inner = max(ma - 2 * lane_half_width, 10)

    ellipse_outer = ((cx, cy), (MA_outer, ma_outer), angle)
    ellipse_inner = ((cx, cy), (MA_inner, ma_inner), angle)
    return ellipse_outer, ellipse_inner


def process_frame(frame, frame_idx, calibrated_ellipse, ellipse_buffer):
    """
    Xử lý 1 frame:
    - Nếu chưa calibrate: phát hiện ellipse, lưu vào buffer.
    - Nếu đã calibrate: vẽ ellipse giữa + OUT/IN line.
    """
    edges = detect_edges_vong8(frame)
    h, w = frame.shape[:2]
    overlay = frame.copy()

    mode_text = ""
    ellipse_mid_to_draw = None
    ellipse_outer = None
    ellipse_inner = None

    if calibrated_ellipse is None:
        # ===== PHASE 1: CALIBRATION =====
        mode_text = f"CALIBRATING {frame_idx+1}/{CALIB_FRAMES}"

        # Tìm contour trên edges
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
            ellipse_mid_to_draw = ellipse_raw

            # tạo OUT/IN tạm cho debug
            ellipse_outer, ellipse_inner = make_out_in_ellipses(
                ellipse_mid_to_draw, LANE_HALF_WIDTH
            )

            # vẽ contour vàng cho debug
            cv2.drawContours(overlay, [best[4]], -1, (0, 255, 255), 1)

        # Sau khi đủ số frame, khóa ellipse
        if (frame_idx + 1) >= CALIB_FRAMES and ellipse_buffer:
            calibrated_ellipse = median_ellipse(ellipse_buffer)
            print(">> CALIB DONE:", calibrated_ellipse)
            mode_text = "RUN (ellipse locked)"
            ellipse_mid_to_draw = calibrated_ellipse
            ellipse_outer, ellipse_inner = make_out_in_ellipses(
                ellipse_mid_to_draw, LANE_HALF_WIDTH
            )

    else:
        # ===== PHASE 2: RUN =====
        mode_text = "RUN (ellipse locked)"
        ellipse_mid_to_draw = calibrated_ellipse
        ellipse_outer, ellipse_inner = make_out_in_ellipses(
            ellipse_mid_to_draw, LANE_HALF_WIDTH
        )

    # Vẽ ellipse nếu có
    if ellipse_mid_to_draw is not None:
        # ellipse giữa: vàng
        cv2.ellipse(overlay, ellipse_mid_to_draw, (0, 255, 255), 1)
    if ellipse_outer is not None:
        # ngoài: đỏ
        cv2.ellipse(overlay, ellipse_outer, (0, 0, 255), 2)
    if ellipse_inner is not None:
        # trong: xanh
        cv2.ellipse(overlay, ellipse_inner, (0, 255, 0), 2)

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
        f"LANE_HALF_WIDTH={LANE_HALF_WIDTH}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    show = cv2.resize(show, None, fx=0.35, fy=0.35)
    return show, calibrated_ellipse, ellipse_buffer


def main():
    calibrated_ellipse = None
    ellipse_buffer = []
    frame_idx = 0

    if USE_IMAGE:
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print("Không đọc được ảnh:", IMAGE_PATH)
            return

        # Với ảnh tĩnh: detect 1 lần, không cần calib nhiều frame
        show, calibrated_ellipse, ellipse_buffer = process_frame(
            frame, 0, None, []
        )
        cv2.imshow("Vong 8 - Image analysis (OUT/IN)", show)
        print("Nhấn phím bất kỳ để thoát...")
        cv2.waitKey(0)

    elif USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("Không mở được video:", VIDEO_PATH)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tạo cửa sổ trước khi tạo trackbar
        cv2.namedWindow("Vong 8 - Video (Calib + Run OUT/IN)", cv2.WINDOW_NORMAL)

        # Hàm callback khi kéo thanh SEEK
        def on_trackbar(pos):
            nonlocal frame_idx, calibrated_ellipse, ellipse_buffer
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            frame_idx = pos
            calibrated_ellipse = None
            ellipse_buffer = []
            print(f">> SEEK TO: frame {pos}")

        # Tạo trackbar (seek bar)
        cv2.createTrackbar(
            "Seek",
            "Vong 8 - Video (Calib + Run OUT/IN)",
            0,
            total_frames - 1,
            on_trackbar,
        )

        while True:
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            show, calibrated_ellipse, ellipse_buffer = process_frame(
                frame, frame_idx, calibrated_ellipse, ellipse_buffer
            )

            # Update slider position (sync UI)
            cv2.setTrackbarPos(
                "Seek", "Vong 8 - Video (Calib + Run OUT/IN)", current_pos
            )

            frame_idx += 1

            cv2.imshow("Vong 8 - Video (Calib + Run OUT/IN)", show)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        cap.release()

    else:
        # Webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không mở được camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            show, calibrated_ellipse, ellipse_buffer = process_frame(
                frame, frame_idx, calibrated_ellipse, ellipse_buffer
            )
            frame_idx += 1

            cv2.imshow("Vong 8 - Webcam (Calib + Run OUT/IN)", show)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
