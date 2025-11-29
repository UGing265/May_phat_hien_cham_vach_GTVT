import cv2
import numpy as np
import math


def detect_edges_vong8(frame):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)

    edges = cv2.Canny(blur, 40, 100)

    # Giữ vùng dưới (vòng 8 nằm nửa dưới ảnh)
    roi_mask = np.zeros_like(edges)
    roi_mask[int(h * 0.4):, :] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    return edges

 
def pick_vong8_candidates(contours, h, w):
    """Lọc contour khả thi là vòng 8."""
    min_area = 3000
    max_area = 0.5 * w * h

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if len(cnt) < 5:
            continue

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)

        # tâm ellipse phải ở dưới
        if cy < h * 0.55:
            continue

        # tỉ lệ trục
        ratio = max(MA, ma) / (min(MA, ma) + 1e-6)
        if ratio < 1.1 or ratio > 4.0:
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


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được camera")
        return

    prev_ellipse = None
    lost_frames = 0            # đếm số frame mất dấu
    max_keep_lost = 5          # giữ ellipse cũ tối đa 5 frame khi mất dấu

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = detect_edges_vong8(frame)

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

        show = cv2.resize(show, None, fx=0.5, fy=0.5)
        cv2.imshow("Vong 8 - Tracking ellipse", show)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
