import cv2
import numpy as np

# ================= CẤU HÌNH =================
IMAGE_PATH = "data/test.jpg"  # đổi lại path ảnh của bạn nếu khác
WHEEL_RADIUS = 15                   # bán kính "bánh xe ảo" (pixel)
TOUCH_THRESH_RATIO = 0.10           # tỉ lệ pixel vạch để coi là chạm
# ============================================


def detect_line_mask(frame):
    """
    Dò vạch sơn vòng 8, trả về ảnh nhị phân line_mask:
    - 255 = pixel thuộc vạch
    -   0 = nền
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tăng tương phản cục bộ để vạch nổi bật hơn
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Top-hat: nhấn các chi tiết sáng mỏng (vạch sơn)
    kernel_size = 81
    kernel_tophat = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel_tophat)

    # Otsu threshold → nhị phân
    _, mask = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Chỉ giữ nửa dưới ảnh (vòng 8)
    roi_mask = np.zeros_like(mask)
    roi_mask[int(h * 0.4):, :] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    # Mượt + CLOSE để nối liền vạch
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


def is_touching_line(cx, cy, line_mask,
                     radius=WHEEL_RADIUS,
                     thresh_ratio=TOUCH_THRESH_RATIO):
    h, w = line_mask.shape[:2]
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return False, 0.0

    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)

    patch = line_mask[y0:y1+1, x0:x1+1]
    if patch.size == 0:
        return False, 0.0

    ph, pw = patch.shape[:2]
    cx_local = pw // 2
    cy_local = ph // 2

    yy, xx = np.ogrid[:ph, :pw]
    circle_mask = (xx - cx_local)**2 + (yy - cy_local)**2 <= radius**2

    white = np.count_nonzero(patch[circle_mask])
    ratio = white / circle_mask.sum()

    return (ratio > thresh_ratio), ratio



def main():
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Không đọc được ảnh:", IMAGE_PATH)
        return

    line_mask = detect_line_mask(frame)

    # Vị trí bánh xe ảo (ban đầu cho ở giữa màn, bạn có thể click để đổi)
    h, w = frame.shape[:2]
    wheel_pos = [w // 2, int(h * 0.7)]

    prev_touch = False
    hit_count = 0

    window_name = "Wheel demo - click de dat banh xe"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Mouse callback: click trái để đặt bánh xe
    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            wheel_pos[0] = x
            wheel_pos[1] = y

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        vis = frame.copy()

        cx, cy = wheel_pos
        touching, ratio = is_touching_line(cx, cy, line_mask)

        # State transition: FREE -> TOUCH
        if touching and not prev_touch:
            hit_count += 1
        prev_touch = touching

        # Vẽ bánh xe ảo
        color = (0, 0, 255) if touching else (0, 255, 0)  # đỏ = chạm, xanh = free
        cv2.circle(vis, (cx, cy), WHEEL_RADIUS, color, 2)
        cv2.circle(vis, (cx, cy), 3, color, -1)

        # Vẽ text debug
        status_text = "TOUCH" if touching else "FREE"
        cv2.putText(
            vis,
            f"STATUS: {status_text}  ratio={ratio:.3f}  hits={hit_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "LEFT CLICK de dat banh xe o vi tri moi",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Hiển thị frame + mask song song
        mask_bgr = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((vis, mask_bgr))

        cv2.imshow(window_name, stacked)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        # optional: nhấn 'r' để reset đếm
        if key in (ord("r"), ord("R")):
            hit_count = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
