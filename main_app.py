# -*- coding: utf-8 -*-
"""
动画序列帧制作 - 主应用

集成所有功能模块，通过顶部导航按钮切换页面：
  1. 视频导帧
  2. 帧数选择
  3. 批量抠图
  4. 色调统一（开发中）
  5. 导出图片（开发中）

运行：
    python main_app.py11
"""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QLabel, QGroupBox, QSizePolicy,
)

from gui_video_to_frames import VideoToFramesPage
from gui_frame_selector import FrameSelectorPage
from gui_frame_matting import FrameMattingPage


# ===================== 占位页面（待开发模块） =====================

class PlaceholderPage(QWidget):
    def __init__(self, title: str, desc: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QWidget{background:#2b2b2b;}")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 28px; font-weight: bold; color: #2d7ff9; background:transparent;")
        title_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(title_lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet("font-size: 15px; color: #bdc3c7; margin-top: 12px; background:transparent;")
        desc_lbl.setAlignment(Qt.AlignCenter)
        desc_lbl.setWordWrap(True)
        lay.addWidget(desc_lbl)

        tip = QLabel("🚧 该功能正在开发中...")
        tip.setStyleSheet("font-size: 13px; color: #7f8c8d; margin-top: 24px; background:transparent;")
        tip.setAlignment(Qt.AlignCenter)
        lay.addWidget(tip)


# ===================== 主窗口 =====================

class MainApp(QMainWindow):
    NAV_ITEMS = [
        ("video",     "视频导帧"),
        ("select",    "帧数选择"),
        ("matting",   "批量抠图"),
        ("color",     "色调统一"),
        ("export",    "导出图片"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("动画序列帧制作工具")
        self.resize(1280, 860)
        self.setMinimumSize(900, 600)

        # 跨页面共享状态
        self.shared_state: dict = {}

        self._build_ui()
        self._switch_page(0)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ===== 顶部导航栏 =====
        nav_bar = QWidget()
        nav_bar.setStyleSheet(
            "QWidget{background:#2c3e50;}"
            "QPushButton{color:#ecf0f1;border:none;padding:14px 28px;"
            "font-size:14px;background:transparent;}"
            "QPushButton:hover{background:#34495e;}"
            "QPushButton:checked{background:#2d7ff9;color:white;font-weight:bold;}")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        # Logo
        logo = QLabel("🎬 动画序列帧制作")
        logo.setStyleSheet(
            "color:white;font-size:16px;font-weight:bold;padding:0 24px;")
        nav_layout.addWidget(logo)
        nav_layout.addStretch()

        self.nav_buttons = []
        self.nav_group = None
        for i, (key, label) in enumerate(self.NAV_ITEMS):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)
        nav_layout.addStretch()

        root.addWidget(nav_bar)

        # ===== 页面堆叠区 =====
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # 页面 1：视频导帧
        self.page_video = VideoToFramesPage(self.shared_state)
        self.stack.addWidget(self.page_video)

        # 页面 2：帧数选择
        self.page_select = FrameSelectorPage(self.shared_state)
        self.stack.addWidget(self.page_select)

        # 页面 3：批量抠图
        self.page_matting = FrameMattingPage(self.shared_state)
        self.stack.addWidget(self.page_matting)

        # 页面 4：色调统一
        self.page_color = PlaceholderPage(
            "色调统一",
            "以原图为基准，统一选中帧的色调（Reinhard 迁移 + 直方图匹配）")
        self.stack.addWidget(self.page_color)

        # 页面 5：导出图片
        self.page_export = PlaceholderPage(
            "导出图片",
            "拼图 / 动画预览 / 分辨率调整 / 导出 PNG 序列或雪碧图")
        self.stack.addWidget(self.page_export)

    def _switch_page(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == idx)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
