import threading
import time

class Detector:
    """
    Class này sau này sẽ chứa:
    - xử lý camera bằng OpenCV
    - phát hiện chạm vạch
    - gọi hàm phát loa khi phát hiện
    """
    def __init__(self):
        self.running = False
        self._thread = None
        self._lock = threading.Lock()

    def _loop(self):
        while True:
            with self._lock:
                if not self.running:
                    break
            # TODO: sau này xử lý frame ở đây
            print("[Detector] Đang xử lý (fake)...")
            time.sleep(1)

    def start(self):
        with self._lock:
            if self.running:
                return
            self.running = True

        print("[Detector] START")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            if not self.running:
                return
            self.running = False
        print("[Detector] STOP")
