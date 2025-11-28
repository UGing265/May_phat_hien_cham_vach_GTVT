from flask import Flask, render_template, jsonify, request

def create_app(detector):
    app = Flask(
        __name__,
        template_folder="ui",
        static_folder="static"
    )

    @app.route("/")
    def index():
        return render_template("index.html")

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

    return app
