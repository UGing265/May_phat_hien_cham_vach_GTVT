# ui/main_window.py
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self._setup_ui()

    def _setup_ui(self):
        # Cấu hình cửa sổ
        self.setWindowTitle("GTvT Detector - UI")
        self.setMinimumSize(800, 500)

        # Layout chính (dọc)
        main_layout = QVBoxLayout()

        # ===== Header =====
        title_label = QLabel("GTvT Detector")
        subtitle_label = QLabel("UI trắng – chuẩn bị gắn camera & OpenCV")

        # Làm chữ tiêu đề nhìn rõ hơn xíu
        title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        subtitle_label.setStyleSheet("font-size: 12px; color: gray;")

        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # Spacer nhỏ
        main_layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # ===== Khu vực chính (sau này để video / thông tin) =====
        center_label = QLabel("Khu vực nội dung chính\n(Sau này sẽ hiển thị camera / thông tin)")
        center_label.setStyleSheet(
            "border: 1px dashed #aaaaaa; "
            "color: #666666; "
            "font-size: 14px;"
        )
        center_label.setMinimumHeight(300)
        center_label.setAlignment(
            center_label.alignment() | 0x0084  # AlignHCenter | AlignVCenter
        )

        main_layout.addWidget(center_label)

        # Spacer nhỏ
        main_layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # ===== Thanh nút điều khiển phía dưới =====
        bottom_layout = QHBoxLayout()

        # Spacer trái
        bottom_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.btn_start = QPushButton("Start (sau này dùng cho camera)")
        self.btn_exit = QPushButton("Thoát")

        bottom_layout.addWidget(self.btn_start)
        bottom_layout.addWidget(self.btn_exit)

        # Spacer phải
        bottom_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(bottom_layout)

        # Gán layout cho window
        self.setLayout(main_layout)

        # Kết nối nút
        self.btn_exit.clicked.connect(self.close)
