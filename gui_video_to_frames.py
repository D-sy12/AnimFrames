# -*- coding: utf-8 -*-
"""
视频导出为序列帧 - 页面（可嵌入主窗口）

依赖: PySide6, opencv-python, numpy
"""

import os
import sys
import json

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QRadioButton, QButtonGroup, QSlider, QProgressBar, QFileDialog,
    QScrollArea, QFrame, QMessageBox, QGroupBox, QSplitter, QSizePolicy,
)

from video_to_frames import extract_frames, FrameInfo


# 会话存档文件（应用根目录）：记录第1页状态与帧目录，
# 程序关闭后重开时恢复上次工作状态，直到用户换视频重新抽帧才被覆盖
SESSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "session.json")


# ===================== 后台抽帧线程 =====================

class ExtractThread(QThread):
    progress = Signal(str)
    finished_ok = Signal(list, str)
    failed = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            import builtins
            orig_print = builtins.print

            def patched_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                self.progress.emit(msg)
                orig_print(*args, **kwargs)

            builtins.print = patched_print
            try:
                saved = extract_frames(**self.params)
            finally:
                builtins.print = orig_print

            self.finished_ok.emit(saved, self.params["output_dir"])
        except Exception as e:
            self.failed.emit(str(e))


# ===================== 页面：视频导帧 =====================

class VideoToFramesPage(QWidget):
    frames_extracted = Signal(str)  # 抽帧完成时发出输出目录

    def __init__(self, shared_state: dict = None, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state if shared_state is not None else {}
        self.thread: ExtractThread | None = None
        self.frames_info: list = []
        self.output_dir: str = ""

        # resize 防抖定时器，避免全屏切换时频繁重建预览
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._do_resize_reload)

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("QGroupBox{border:none;margin-top:8px;} QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(0)

        self.h_layout = QHBoxLayout()
        self.h_layout.setSpacing(8)
        root.addLayout(self.h_layout, 1)

        # ========== 左侧：导出图片预览 ==========
        self.preview_group = QGroupBox("抽取结果预览")
        self.preview_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        prg = QVBoxLayout(self.preview_group)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setMinimumSize(400, 300)
        self.scroll.setStyleSheet("QScrollArea{background:#2b2b2b;border:1px solid #444;}")
        self.thumb_container = QWidget()
        self.thumb_container.setStyleSheet("QWidget{background:#2b2b2b;}")
        self.thumb_layout = QGridLayout(self.thumb_container)
        self.thumb_layout.setSpacing(8)
        self.scroll.setWidget(self.thumb_container)
        prg.addWidget(self.scroll, 1)
        self.h_layout.addWidget(self.preview_group, 13)

        # ========== 右侧：所有功能控件 ==========
        right_panel = QWidget()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 视频文件
        video_group = QGroupBox("视频文件")
        vg = QHBoxLayout(video_group)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择要处理的视频文件...")
        btn_browse_video = QPushButton("浏览...")
        btn_browse_video.clicked.connect(self._browse_video)
        vg.addWidget(self.video_path_edit, 1)
        vg.addWidget(btn_browse_video)
        right_layout.addWidget(video_group)

        # 抽帧参数
        params_group = QGroupBox("抽帧参数")
        pg = QGridLayout(params_group)
        pg.setHorizontalSpacing(16)
        pg.setVerticalSpacing(8)

        pg.addWidget(QLabel("抽帧模式:"), 0, 0)
        self.mode_all = QRadioButton("逐帧")
        self.mode_interval = QRadioButton("每 N 帧")
        self.mode_time = QRadioButton("每 N 秒")
        self.mode_all.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_all, 0)
        self.mode_group.addButton(self.mode_interval, 1)
        self.mode_group.addButton(self.mode_time, 2)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_all)
        mode_row.addWidget(self.mode_interval)
        mode_row.addWidget(self.mode_time)
        mode_row.addStretch()
        pg.addLayout(mode_row, 0, 1, 1, 3)

        pg.addWidget(QLabel("间隔:"), 1, 0)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.01, 10000)
        self.interval_spin.setValue(1)
        self.interval_spin.setDecimals(2)
        self.interval_label = QLabel("帧")
        self.interval_spin.setEnabled(False)
        pg.addWidget(self.interval_spin, 1, 1)
        pg.addWidget(self.interval_label, 1, 2)
        pg.addWidget(QWidget(), 1, 3)

        pg.addWidget(QLabel("起始时间(秒):"), 2, 0)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 99999)
        self.start_spin.setValue(0)
        self.start_spin.setDecimals(2)
        pg.addWidget(self.start_spin, 2, 1)

        pg.addWidget(QLabel("结束时间(秒):"), 2, 2)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 99999)
        self.end_spin.setValue(0)
        self.end_spin.setDecimals(2)
        self.end_spin.setSpecialValueText("不限制")
        pg.addWidget(self.end_spin, 2, 3)

        pg.addWidget(QLabel("输出格式:"), 3, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg"])
        pg.addWidget(self.format_combo, 3, 1)

        pg.addWidget(QLabel("文件名前缀:"), 3, 2)
        self.prefix_edit = QLineEdit("frame")
        pg.addWidget(self.prefix_edit, 3, 3)
        right_layout.addWidget(params_group)

        # 去重
        dedup_group = QGroupBox("相同帧去重")
        dg = QGridLayout(dedup_group)
        self.dedup_check = QCheckBox("启用去重（相似帧只保留一张）")
        self.dedup_check.setChecked(True)
        dg.addWidget(self.dedup_check, 0, 0, 1, 4)
        dg.addWidget(QLabel("相似度阈值:"), 1, 0)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(95)
        self.threshold_label = QLabel("95 %")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v} %"))
        dg.addWidget(self.threshold_slider, 1, 1)
        dg.addWidget(self.threshold_label, 1, 2)
        tip = QLabel("（数值越高越严格）")
        tip.setStyleSheet("color: #999; font-size: 11px;")
        dg.addWidget(tip, 1, 3)
        right_layout.addWidget(dedup_group)

        # 输出
        out_group = QGroupBox("输出")
        og = QHBoxLayout(out_group)
        og.addWidget(QLabel("输出目录:"))
        self.output_edit = QLineEdit("frames")
        btn_browse_out = QPushButton("浏览...")
        btn_browse_out.clicked.connect(self._browse_output)
        og.addWidget(self.output_edit, 1)
        og.addWidget(btn_browse_out)
        right_layout.addWidget(out_group)

        # 操作按钮
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始抽帧")
        self.btn_start.setMinimumHeight(36)
        self.btn_start.setStyleSheet(
            "QPushButton{background:#2d7ff9;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#1a6fe0;}"
            "QPushButton:disabled{background:#aaa;}")
        self.btn_start.clicked.connect(self._start)
        btn_row.addWidget(self.btn_start, 2)

        self.btn_open_dir = QPushButton("打开输出目录")
        self.btn_open_dir.clicked.connect(self._open_output_dir)
        btn_row.addWidget(self.btn_open_dir, 1)
        right_layout.addLayout(btn_row)

        # 抽取结果信息
        self.summary_label = QLabel("尚未抽取")
        self.summary_label.setStyleSheet("color: #2d7ff9; font-weight: bold;")
        self.summary_label.setWordWrap(True)
        right_layout.addWidget(self.summary_label)

        # 进度
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #555;")
        right_layout.addWidget(self.status_label)

        right_layout.addStretch()
        # 右侧面板包裹滚动区，高缩放下内容可滚动
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("QScrollArea{background:#2b2b2b;border:none;}")
        right_scroll.setWidget(right_panel)
        self.h_layout.addWidget(right_scroll, 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frames_info:
            self._resize_timer.start()

    def _do_resize_reload(self):
        if self.frames_info:
            self._load_preview()

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*.*)")
        if path:
            self.video_path_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    def _open_output_dir(self):
        d = self.output_edit.text().strip()
        if d and os.path.isdir(d):
            os.startfile(os.path.abspath(d))
        else:
            QMessageBox.warning(self, "提示", "输出目录不存在，请先抽取帧。")

    def _on_mode_changed(self):
        mode = self.mode_group.checkedId()
        if mode == 0:
            self.interval_spin.setEnabled(False)
            self.interval_label.setText("—")
        elif mode == 1:
            self.interval_spin.setEnabled(True)
            self.interval_spin.setDecimals(0)
            self.interval_spin.setRange(1, 100000)
            self.interval_label.setText("帧")
        else:
            self.interval_spin.setEnabled(True)
            self.interval_spin.setDecimals(2)
            self.interval_spin.setRange(0.01, 10000)
            self.interval_label.setText("秒")

    def _start(self):
        video = self.video_path_edit.text().strip()
        if not video:
            QMessageBox.warning(self, "提示", "请先选择视频文件！")
            return
        if not os.path.isfile(video):
            QMessageBox.warning(self, "提示", f"视频文件不存在:\n{video}")
            return

        out_dir = self.output_edit.text().strip() or "frames"
        mode_id = self.mode_group.checkedId()
        mode = ["all", "interval", "time"][mode_id]
        end_time = None if self.end_spin.value() == 0 else self.end_spin.value()

        params = dict(
            video_path=video, output_dir=out_dir, mode=mode,
            interval=self.interval_spin.value(),
            dedup=self.dedup_check.isChecked(),
            similarity_threshold=float(self.threshold_slider.value()),
            img_format=self.format_combo.currentText(),
            prefix=self.prefix_edit.text().strip() or "frame",
            start_time=self.start_spin.value(), end_time=end_time,
        )

        self._clear_preview()
        self.progress_bar.setValue(0)
        self.status_label.setText("正在抽取...")
        self.btn_start.setEnabled(False)

        self.thread = ExtractThread(params)
        self.thread.progress.connect(self._on_progress)
        self.thread.finished_ok.connect(self._on_finished)
        self.thread.failed.connect(self._on_failed)
        self.thread.start()

    def _on_progress(self, msg: str):
        self.status_label.setText(msg)
        if msg.startswith("[进度]"):
            try:
                parts = msg.split()
                cur = int(parts[1].split("/")[0])
                total = int(parts[1].split("/")[1])
                if total > 0:
                    self.progress_bar.setValue(int(cur / total * 100))
            except Exception:
                pass

    def _on_finished(self, saved_frames: list, output_dir: str):
        self.frames_info = saved_frames
        self.output_dir = output_dir
        self.btn_start.setEnabled(True)
        self.progress_bar.setValue(100)

        info_path = os.path.join(output_dir, "frames_info.json")
        total = skipped = 0
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                total = data.get("exported_count", len(saved_frames))
                skipped = data.get("skipped_duplicates", 0)
            except Exception:
                total = len(saved_frames)

        self.summary_label.setText(
            f"抽取完成：总帧数 {total + skipped}  |  保存 {total} 帧  |  跳过重复 {skipped} 帧  →  "
            f"{os.path.abspath(output_dir)}")
        self.status_label.setText("完成")
        self._load_preview()

        # 共享输出目录给其他页面
        self.shared_state["frames_dir"] = os.path.abspath(output_dir)
        self.shared_state["frames_info"] = saved_frames
        # 重新抽帧后旧的选中列表失效，清除避免后续模块读到残留索引
        self.shared_state.pop("selected_indices", None)
        self.shared_state.pop("selected_files", None)
        self.shared_state.pop("selected_frames", None)
        # 旧视频的抠图/精抠目录已在抽帧时删除，清除指向它们的路径，
        # 让批量抠图、图片精抠页进入时重新加载为空状态
        self.shared_state.pop("matted_dir", None)
        self.shared_state.pop("refined_dir", None)
        self.shared_state.pop("colored_dir", None)
        self.shared_state.pop("adjusted_dir", None)
        self._save_session()
        self.frames_extracted.emit(output_dir)

    # ---------------- 会话存档 ----------------

    def _save_session(self):
        """抽帧完成后保存第1页状态，供下次启动恢复"""
        if not self.thread:
            return
        p = self.thread.params
        data = {
            "video_path": os.path.abspath(p["video_path"]),
            "frames_dir": os.path.abspath(p["output_dir"]),
            "mode": p["mode"],
            "interval": p["interval"],
            "start_time": p["start_time"],
            "end_time": p["end_time"],
            "img_format": p["img_format"],
            "prefix": p["prefix"],
            "dedup": p["dedup"],
            "similarity_threshold": p["similarity_threshold"],
        }
        try:
            with open(SESSION_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def restore_session(self, data: dict) -> bool:
        """从存档恢复第1页（回填参数 + 加载已有帧，不重新抽帧）"""
        video = data.get("video_path", "")
        frames_dir = data.get("frames_dir", "")
        info_path = os.path.join(frames_dir, "frames_info.json")
        if not (video and os.path.isfile(video)):
            return False
        if not os.path.isfile(info_path):
            return False
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        # 回填控件
        self.video_path_edit.setText(video)
        self.output_edit.setText(frames_dir)
        mode_id = {"all": 0, "interval": 1, "time": 2}.get(
            data.get("mode", "all"), 0)
        self.mode_group.button(mode_id).setChecked(True)
        self._on_mode_changed()
        self.interval_spin.setValue(data.get("interval", 1))
        self.start_spin.setValue(data.get("start_time", 0))
        end_time = data.get("end_time")
        self.end_spin.setValue(end_time if end_time else 0)
        idx = self.format_combo.findText(data.get("img_format", "png"))
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.prefix_edit.setText(data.get("prefix", "frame"))
        self.dedup_check.setChecked(data.get("dedup", True))
        self.threshold_slider.setValue(
            int(data.get("similarity_threshold", 95)))

        # 重建帧列表与预览（仅非重复帧有文件）
        self.output_dir = frames_dir
        self.frames_info = [
            FrameInfo(**fr) for fr in info_data.get("frames", [])
            if not fr.get("is_duplicate")]
        total = info_data.get("exported_count", len(self.frames_info))
        skipped = info_data.get("skipped_duplicates", 0)
        self.summary_label.setText(
            f"已恢复上次会话：保存 {total} 帧  |  跳过重复 {skipped} 帧  →  "
            f"{os.path.abspath(frames_dir)}")
        self.status_label.setText("已恢复上次的工作进度")
        self.progress_bar.setValue(100)
        self._load_preview()

        self.shared_state["frames_dir"] = os.path.abspath(frames_dir)
        self.shared_state["frames_info"] = self.frames_info
        self.shared_state["video_path"] = video
        return True

    def _on_failed(self, err: str):
        self.btn_start.setEnabled(True)
        self.status_label.setText("失败")
        QMessageBox.critical(self, "出错了", err)

    def _clear_preview(self):
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _load_preview(self):
        self._clear_preview()
        if not self.frames_info:
            return
        thumb_w = 120
        thumb_h = 90
        cell_w = thumb_w + 12 + 8  # 缩略图宽 + padding + spacing
        avail = self.scroll.viewport().width()
        cols = max(1, avail // cell_w) if avail > 0 else 4
        for i, info in enumerate(self.frames_info):
            path = os.path.join(self.output_dir, info.filename)
            if not os.path.isfile(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            pix = pix.scaled(thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            cell = QFrame()
            cell.setFrameShape(QFrame.StyledPanel)
            cell.setFixedWidth(thumb_w + 12)
            cell.setStyleSheet(
                "QFrame{border: 1px solid #444; border-radius: 4px; "
                "background: #1e1e1e;}")
            lay = QVBoxLayout(cell)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(2)
            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)
            lay.addWidget(img_label)
            cap = QLabel(f"#{info.index}\n{info.timestamp_ms:.0f}ms")
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("font-size: 10px; color: #999;")
            lay.addWidget(cap)
            r, c = divmod(i, cols)
            self.thumb_layout.addWidget(cell, r, c)
        self.thumb_layout.setRowStretch(self.thumb_layout.rowCount(), 1)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = QMainWindow()
    win.setWindowTitle("视频导出序列帧")
    win.resize(1100, 750)
    page = VideoToFramesPage()
    win.setCentralWidget(page)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
