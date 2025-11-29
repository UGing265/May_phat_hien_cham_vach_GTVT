import cv2
import numpy as np

def detect_edges_vong8(frame):
    """
    Trả về ảnh edges (Canny) tập trung ở vùng vòng số 8.
    - Dùng Canny để lấy biên
    - Chỉ giữ vùng nửa dưới ảnh (nơi có vạch 8)
    """
    h, w = frame.shape[:2]

    # 1. Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur nhẹ để giảm nhiễu (thử đổi (5,5), (7,7) nếu cần)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)

    # 3. Canny edge (2 ngưỡng bạn có thể chỉnh tay để hợp mắt)
    edges = cv2.Canny(blur, 50, 150)

    # 4. Chỉ giữ vùng dưới (0.4 * chiều cao trở xuống) – nơi có vòng 8
    roi_mask = np.zeros_like(edges)
    roi_mask[int(h * 0.4):, :] = 255          # từ 40% chiều cao trở xuống là ROI
    edges = cv2.bitwise_and(edges, roi_mask)

    return edges


def main():
    # ⬇️ Nếu muốn test webcam:
    cap = cv2.VideoCapture(0)

    # ⬇️ Nếu muốn test 1 file ảnh cố định (ảnh vòng số 8 của bạn):
    # frame = cv2.imread("photo_2025-11-29_06-23-52.jpg")
    # edges = detect_edges_vong8(frame)
    # overlay = frame.copy()
    # overlay[edges != 0] = (0, 255, 0)
    # show = np.hstack((frame, overlay, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
    # cv2.imshow("Canny vong 8", show)
    # cv2.waitKey(0)
    # return

    if not cap.isOpened():
        print("Không mở được camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = detect_edges_vong8(frame)

        # Tạo overlay: chỗ nào có edge thì tô xanh
        overlay = frame.copy()
        overlay[edges != 0] = (0, 255, 0)

        # Ghép 3 ảnh để dễ so sánh: gốc | overlay | edges
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        show = np.hstack((frame, overlay, edges_bgr))

        cv2.imshow("Canny Vong 8 Demo (goc | overlay | edges)", show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
