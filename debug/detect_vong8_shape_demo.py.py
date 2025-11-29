import cv2
import numpy as np


def detect_edges_vong8(frame):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)

    edges = cv2.Canny(blur, 50, 150)

    # Giữ vùng dưới (vòng 8 nằm nửa dưới ảnh)
    roi_mask = np.zeros_like(edges)
    roi_mask[int(h * 0.4):, :] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    return edges


def pick_vong8_candidates(contours, h, w):
    """
    Lọc contour nào có khả năng là vòng số 8:
    - Diện tích trong khoảng cho phép
    - Tâm nằm phía dưới
    - Tỉ lệ trục ellipse hợp lý (không quá tròn, không quá dẹt)
    """
    min_area = 3000
    max_area = 0.5 * w * h  # không cho quá to bằng nửa màn hình

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if len(cnt) < 5:
            continue

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)

        # Tâm ellipse phải nằm ở nửa dưới (gần mặt đất hơn)
        if cy < h * 0.55:
            continue

        # Tỉ lệ 2 trục
        ratio = max(MA, ma) / (min(MA, ma) + 1e-6)
        if ratio < 1.1 or ratio > 4.0:
            continue

        candidates.append(((cx, cy), (MA, ma), angle, area, cnt))

    return candidates


def smooth_ellipse(new_ellipse, prev_ellipse, alpha=0.3):
    """
    Làm mượt ellipse mới với ellipse cũ bằng EMA (Exponential Moving Average)
    alpha càng nhỏ → chuyển động càng mượt nhưng chậm.
    """
    if prev_ellipse is None:
        return new_ellipse

    (cx, cy), (MA, ma), angle = new_ellipse
    (pcx, pcy), (pMA, pma), pangle = prev_ellipse

    sx = alpha * cx + (1 - alpha) * pcx
    sy = alpha * cy + (1 - alpha) * pcy
    sMA = alpha * MA + (1 - alpha) * pMA
    sma = alpha * ma + (1 - alpha) * pma
    sangle = alpha * angle + (1 - alpha) * pangle

    return ( (sx, sy), (sMA, sma), sangle )


def main():
    cap = cv2.VideoCapture(0)  # hoặc đọc từ file nếu muốn: cv2.VideoCapture("video.mp4")

    if not cap.isOpened():
        print("Không mở được camera")
        return

    prev_ellipse = None  # lưu ellipse của frame trước

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = detect_edges_vong8(frame)

        # 1) Nối các đoạn edge bằng morphology CLOSE
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2) Tìm contour từ edge đã nối
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # --- Lọc contour ứng viên vòng 8 ---
        candidates = pick_vong8_candidates(contours, h, w)

        ellipse_to_draw = None

        if candidates:
            # Nếu đã có ellipse cũ → ưu tiên contour gần ellipse cũ nhất
            if prev_ellipse is not None:
                pcx, pcy = prev_ellipse[0]
                candidates.sort(
                    key=lambda c: (c[0][0] - pcx) ** 2 + (c[0][1] - pcy) ** 2
                )
            else:
                # Chưa có ellipse cũ → chọn theo diện tích lớn nhất
                candidates.sort(key=lambda c: c[3], reverse=True)

            best = candidates[0]
            ellipse_raw = (best[0], best[1], best[2])

            # Làm mượt với ellipse cũ
            ellipse_to_draw = smooth_ellipse(ellipse_raw, prev_ellipse, alpha=0.3)
            prev_ellipse = ellipse_to_draw

            # Vẽ contour ứng viên (màu vàng) để debug
            cv2.drawContours(overlay, [best[4]], -1, (0, 255, 255), 1)

        else:
            # Không tìm được contour phù hợp:
            # có thể giữ nguyên ellipse cũ 1–2 frame cho đỡ giật
            ellipse_to_draw = prev_ellipse

        # Vẽ ellipse màu xanh nếu có
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

        # Thu nhỏ cho vừa màn
        scale = 0.5
        show = cv2.resize(show, None, fx=scale, fy=scale)

        cv2.imshow("Vong 8 - Canny + Ellipse fit", show)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
