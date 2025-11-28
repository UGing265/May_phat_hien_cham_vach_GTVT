# server/api.py

import os
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app(detector):
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "ui"),
        static_folder=os.path.join(BASE_DIR, "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    # ==== API điều khiển ====

    @app.route("/api/status")
    def status():
        return jsonify({"running": detector.running})

    @app.route("/api/start", methods=["POST"])
    def start():
        detector.start()
        return jsonify({"ok": True, "running": detector.running})

    @app.route("/api/stop", methods=["POST"])
    def stop():
        detector.stop()
        return jsonify({"ok": True, "running": detector.running})

    @app.route("/api/play-sound", methods=["POST"])
    def play_sound():
        # TODO: sau này gọi code phát loa trên Raspberry Pi
        print('[SOUND] "Xe số _ chạm vạch!" (fake)')
        return jsonify({"ok": True, "message": "Đã phát thử âm thanh (fake)"})

    # ==== API nhận frame từ iPhone ====

    @app.route("/api/upload-frame", methods=["POST"])
    def upload_frame():
        """
        iPhone sẽ POST 1 hình JPEG (frame) lên đây (FormData field 'frame').
        Mình decode bằng OpenCV rồi đưa cho detector.process_frame().
        """
        file = request.files.get("frame")
        if not file:
            return jsonify({"ok": False, "error": "No frame"}), 400

        data = file.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"ok": False, "error": "Decode failed"}), 400

        detector.process_frame(frame)
        return jsonify({"ok": True})

    return app
