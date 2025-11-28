from core.detector import Detector
from server.api import create_app

def main():
    detector = Detector()
    app = create_app(detector)

    # host='0.0.0.0' để iPhone trong cùng mạng truy cập được
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
