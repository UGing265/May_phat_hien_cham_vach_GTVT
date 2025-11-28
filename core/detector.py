# core/detector.py

import threading

class Detector:
    """
    Detector nhận frame từ iPhone và xử lý.
    - start(): bật chế độ xử lý
    - stop(): tắt xử lý
    - process_frame(frame): xử lý từng frame (numpy array BGR)
    """

    def __init__(self):
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        # Bật cờ đang chạy
        with self._lock:
            self.running = True
        print("[Detector] START (sẽ xử lý frame từ iPhone)")

    def stop(self):
        # Tắt cờ đang chạy
        with self._lock:
            self.running = False
        print("[Detector] STOP (ngưng xử lý frame)")

    def process_frame(self, frame):
        """
        frame: numpy array (BGR) từ iPhone upload.
        Ở đây CHƯA làm chạm vạch, chỉ log để biết pipeline chạy.
        Sau này bạn thêm OpenCV ở đây.
        """
        with self._lock:
            if not self.running:
                # Nếu chưa bấm Start thì bỏ qua frame
                return

        # TODO: sau này bạn đặt code OpenCV: detect line, detect wheel, v.v.
        h, w, _ = frame.shape
        print(f"[Detector] Nhận frame {w}x{h} từ iPhone (running = True)")
