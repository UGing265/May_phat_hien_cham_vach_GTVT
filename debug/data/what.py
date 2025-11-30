import cv2
import numpy as np

# Kích thước khung hình
W, H = 800, 600

# Thông số ellipse (giả vòng 8)
cx, cy = W // 2, H // 2      # tâm ellipse
a, b = 250, 150              # bán trục ngang / dọc

def is_inside_ellipse(x, y, cx, cy, a, b):
    # Trả về True nếu (x, y) nằm TRONG ellipse
    val = ((x - cx) ** 2) / (a ** 2) + ((y - cy) ** 2) / (b ** 2)
    return val <= 1.0

def main():
    prev_inside = None
    hit_count = 0

    # Điểm bánh xe giả lập: chạy từ trên xuống dưới, cắt qua ellipse
    x = cx
    y_start = cy - b - 50      # bắt đầu phía trên ellipse
    y_end   = cy + b + 50      # kết thúc phía dưới ellipse

    for y in range(y_start, y_end + 1):
        # Tạo frame đen
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Vẽ ellipse trắng (vòng 8)
        cv2.ellipse(frame, (cx, cy), (a, b), 0, 0, 360, (255, 255, 255), 2)

        # Vẽ điểm bánh xe (màu xanh)
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

        # Kiểm tra trong/ngoài ellipse
        inside = is_inside_ellipse(x, y, cx, cy, a, b)

        # Vẽ text trạng thái
        state_text = "INSIDE" if inside else "OUTSIDE"
        cv2.putText(frame, f"State: {state_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        # Nếu frame trước là INSIDE, frame này OUTSIDE -> CHẠM VẠCH
        if prev_inside is True and inside is False:
            hit_count += 1
            print(f"CHAM VACH tại y = {y}")
            cv2.putText(frame, "CHAM VACH!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        prev_inside = inside

        cv2.imshow("Demo cham vach", frame)
        key = cv2.waitKey(15)
        if key == 27:  # ESC để thoát
            break

    print("So lan cham vach:", hit_count)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
