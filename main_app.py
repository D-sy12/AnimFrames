# -*- coding: utf-8 -*-
"""
动画序列帧制作 - 主应用

集成所有功能模块，通过顶部导航按钮切换页面：
  1. 视频导帧
  2. 帧数选择
  3. 批量抠图
  4. 图片精抠
  5. 色调统一
  6. 调整位置
  7. 导出图片
  8. 动画预览（独立工具，不参与会话存档）

运行：
    python main_app.py11
"""

import sys
import os
import json

from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QLabel, QGroupBox, QSizePolicy,
    QAbstractSpinBox,
)


def _resource_path(name: str) -> str:
    """资源文件路径：开发时取脚本目录，PyInstaller 打包后取 _MEIPASS"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


class _SpinFocusCleaner(QObject):
    """点击数值框以外的位置时退出其编辑状态。

    Qt 默认行为：数值框（QSpinBox/QDoubleSpinBox）获得焦点后，点击其他
    地方不会失去焦点，键盘输入会一直留在框内。此过滤器在全局鼠标按下时
    检查：若焦点在数值框内而点击目标不在其中，则清除焦点（提交当前值）。
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            w = QApplication.focusWidget()
            while w is not None and not isinstance(w, QAbstractSpinBox):
                w = w.parentWidget()          # 焦点常在内部 lineEdit 上
            if w is None:
                return False
            # 事件目标可能是 QWindow（点窗口框架/原生对话框），非 QWidget
            if not isinstance(obj, QWidget) or not w.isAncestorOf(obj):
                w.clearFocus()
        return False

from gui_video_to_frames import VideoToFramesPage, SESSION_PATH
from gui_frame_selector import FrameSelectorPage
from gui_frame_matting import FrameMattingPage
from gui_frame_refine import FrameRefinePage
from gui_frame_color import FrameColorPage
from gui_frame_adjust import FrameAdjustPage
from gui_frame_export import FrameExportPage
from gui_anim_preview import AnimPreviewPage


# ===================== 主窗口 =====================

class MainApp(QMainWindow):
    NAV_ITEMS = [
        ("video",     "视频导帧"),
        ("select",    "帧数选择"),
        ("matting",   "批量抠图"),
        ("refine",    "图片精抠"),
        ("color",     "色调统一"),
        ("adjust",    "调整位置"),
        ("export",    "导出图片"),
        ("anim",      "动画预览"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("动画序列帧制作工具")
        self.resize(1280, 860)
        self.setMinimumSize(900, 600)

        # 跨页面共享状态
        self.shared_state: dict = {}

        self._build_ui()
        self._restore_session()
        self._switch_page(0)

    def _restore_session(self):
        """启动时恢复上次会话：只要不换视频，重开程序回到离开时的状态。

        各下游页面的产物（selected_frames/matted/refined/colored/adjusted
        及各自 info json）本就持久化在帧目录里，这里只需重建 shared_state
        并让第1页回填，后续页面进入时会自行从磁盘恢复。
        """
        try:
            with open(SESSION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if not self.page_video.restore_session(data):
            return
        # 探测下游产物目录，存在则挂到共享状态
        frames_dir = self.shared_state.get("frames_dir", "")
        for key, sub in (("matted_dir", "matted"), ("refined_dir", "refined"),
                         ("colored_dir", "colored"), ("adjusted_dir", "adjusted")):
            d = os.path.join(frames_dir, sub)
            if os.path.isdir(d):
                self.shared_state[key] = d

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

        # 页面 4：图片精抠
        self.page_refine = FrameRefinePage(self.shared_state)
        self.stack.addWidget(self.page_refine)

        # 页面 5：色调统一
        self.page_color = FrameColorPage(self.shared_state)
        self.stack.addWidget(self.page_color)

        # 页面 6：调整位置
        self.page_adjust = FrameAdjustPage(self.shared_state)
        self.stack.addWidget(self.page_adjust)

        # 页面 7：导出图片
        self.page_export = FrameExportPage(self.shared_state)
        self.stack.addWidget(self.page_export)

        # 页面 8：动画预览（独立工具，不参与会话存档）
        self.page_anim = AnimPreviewPage(self.shared_state)
        self.stack.addWidget(self.page_anim)

    def _switch_page(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == idx)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon_path = _resource_path("app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    cleaner = _SpinFocusCleaner(app)     # 点击数值框外部即退出其编辑状态
    app.installEventFilter(cleaner)
    win = MainApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
