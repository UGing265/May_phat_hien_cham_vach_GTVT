import cv2
import numpy as np

# Đổi sang 0 nếu muốn test webcam
VIDEO_PATH = "data/IMG_1242.MOV"
WINDOW_NAME = "PA1 - TopHat SafeArea Demo"

def build_safe_area(frame):
    """
    Input: 1 frame BGR
    Output:
        - safe_mask: mask vùng an toàn (0/255)
        - debug: dict ảnh debug
    """
    debug = {}

    # 1) Gray + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    debug["gray"] = gray.copy()

    # 2) TopHat làm nổi vạch trắng
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    debug["tophat"] = tophat.copy()

    # 3) Lọc vạch
    tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    _, lines = cv2.threshold(tophat_norm, 40, 255, cv2.THRESH_BINARY)
    debug["lines_raw"] = lines.copy()

    # 4) Nối các đoạn vạch
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    lines_closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    debug["lines_closed"] = lines_closed.copy()

    # 5) Distance transform
    inv = cv2.bitwise_not(lines_closed)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    debug["dist_vis"] = dist_vis.copy()

    # 6) Ngưỡng theo tỉ lệ max distance
    dmax = dist.max()
    # tuỳ video, nhưng thường 0.25–0.7 là ổn
    low = 0.25 * dmax
    high = 0.70 * dmax

    mid_region = ((dist > low) & (dist < high)).astype("uint8") * 255

    # làm mịn một chút
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mid_region = cv2.morphologyEx(mid_region, cv2.MORPH_OPEN, kernel_open, iterations=1)
    debug["mid_region"] = mid_region.copy()

    # 7) Chỉ giữ 1 component: cái đi qua vùng giữa màn hình
    h, w = mid_region.shape[:2]
    num_labels, labels = cv2.connectedComponents(mid_region)

    safe_mask = np.zeros_like(mid_region)

    # anchor: điểm ở khoảng 3/4 chiều cao, giữa màn hình (nơi đường vòng 8 chạy)
    anchor_y = int(h * 0.7)
    anchor_x = int(w * 0.5)

    anchor_label = labels[anchor_y, anchor_x]
    if anchor_label != 0:
        safe_mask[labels == anchor_label] = 255
    else:
        # nếu anchor rơi vào nền (0) thì fallback: chọn contour lớn nhất
        contours, _ = cv2.findContours(mid_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(safe_mask, [c], -1, 255, thickness=-1)

    debug["safe_mask"] = safe_mask.copy()
    return safe_mask, debug


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Khong mo duoc video/camera")
        return

    first_safe_mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(20) & 0xFF

        # Nhấn R để reset safe area
        if key == ord('r'):
            print("RESET SAFE AREA!")
            first_safe_mask = None
            continue

        # Nhấn Q để thoát
        if key == ord('q'):
            break

        # Nếu chưa có safe area → tạo lần đầu
        if first_safe_mask is None:
            safe_mask, debug = build_safe_area(frame)
            first_safe_mask = safe_mask
        else:
            safe_mask = first_safe_mask

        # Overlay safe mask lên frame
        overlay = frame.copy()
        overlay[safe_mask == 255] = (0, 255, 0)

        blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        cv2.imshow(WINDOW_NAME, blended)


        # Nếu muốn xem debug từng bước thì mở thêm:
        # if "tophat" in debug: cv2.imshow("tophat", debug["tophat"])
        # if "lines_closed" in debug: cv2.imshow("lines_closed", debug["lines_closed"])
        # if "dist_vis" in debug: cv2.imshow("dist_vis", debug["dist_vis"])
        # if "safe_mask" in debug: cv2.imshow("safe_mask", debug["safe_mask"])

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
