# -*- coding: utf-8 -*-
"""
模块二：帧二次筛选 + 动画预览 - 页面（可嵌入主窗口）

选中帧=彩色高亮，未选中帧=灰度变暗
播放模式：循环 / 来回 / 一次性
"""

import os
import sys
import json
from typing import List

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QSlider, QFileDialog, QScrollArea,
    QFrame, QMessageBox, QGroupBox, QSizePolicy, QSplitter, QRadioButton,
    QButtonGroup, QCheckBox,
)

import numpy as np


# ===================== 帧数据结构 =====================

class FrameItem:
    def __init__(self, index: int, filename: str, filepath: str,
                 original_index: int = -1, timestamp_ms: float = 0.0):
        self.index = index
        self.filename = filename
        self.filepath = filepath
        self.original_index = original_index
        self.timestamp_ms = timestamp_ms
        self.selected = False


# ===================== 缩略图单元格 =====================

def make_gray_pixmap(pix: QPixmap) -> QPixmap:
    img = pix.toImage().convertToFormat(QImage.Format_Grayscale8)
    img = img.convertToFormat(QImage.Format_ARGB32)
    for y in range(img.height()):
        line = img.scanLine(y)
        arr = np.frombuffer(line, dtype=np.uint8).reshape(img.width(), 4)
        arr[:, 0:3] = (arr[:, 0:3].astype(np.float32) * 0.55).astype(np.uint8)
    return QPixmap.fromImage(img)


class ThumbCell(QFrame):
    check_toggled = Signal(int, bool)   # 左上角选择框勾选变化（index, checked）

    def __init__(self, index: int, filepath: str, thumb_w: int = 110, thumb_h: int = 80):
        super().__init__()
        self.index = index
        self.filepath = filepath
        self._selected = False
        self._current = False

        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(thumb_w + 14)
        self.setFixedHeight(thumb_h + 32)
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(thumb_w, thumb_h)
        pix = QPixmap(filepath)
        if not pix.isNull():
            pix = pix.scaled(thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._orig_pix = pix
        self._gray_pix = make_gray_pixmap(pix) if not pix.isNull() else pix
        self.img_label.setPixmap(self._gray_pix)
        lay.addWidget(self.img_label, 0, Qt.AlignHCenter)

        self.caption = QLabel(f"#{index + 1}")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setFixedHeight(16)
        self.caption.setStyleSheet("font-size: 10px; color: #999;")
        lay.addWidget(self.caption, 0, Qt.AlignHCenter)

        # 左上角选择框：勾选=选中（点击图片本体仅预览，不改变选中）
        self.chk = QCheckBox(self)
        self.chk.setToolTip("勾选=选中该帧")
        self.chk.setStyleSheet(
            "QCheckBox{background:rgba(20,20,20,190);border-radius:3px;"
            "padding:1px;spacing:0px;}"
            "QCheckBox::indicator{width:15px;height:15px;}")
        self.chk.move(7, 7)
        self.chk.raise_()
        self.chk.toggled.connect(
            lambda on: self.check_toggled.emit(self.index, bool(on)))

        self._update_style()

    def set_selected(self, sel: bool):
        self._selected = sel
        if self.chk.isChecked() != sel:      # 程序化改动同步勾选框（不递归）
            self.chk.blockSignals(True)
            self.chk.setChecked(sel)
            self.chk.blockSignals(False)
        self._update_style()

    def set_current(self, cur: bool):
        self._current = cur
        self._update_style()

    def is_selected(self):
        return self._selected

    def _update_style(self):
        # 当前播放帧：黄色边框优先
        if self._current:
            if self._selected:
                self.img_label.setPixmap(self._orig_pix)
            else:
                self.img_label.setPixmap(self._gray_pix)
            self.setStyleSheet(
                "QFrame{border: 3px solid #f1c40f; border-radius: 4px; "
                "background: #3a3a1e;}")
            self.caption.setStyleSheet("font-size: 10px; color: #f1c40f; font-weight: bold;")
        elif self._selected:
            self.img_label.setPixmap(self._orig_pix)
            self.setStyleSheet(
                "QFrame{border: 3px solid #2d7ff9; border-radius: 4px; "
                "background: #1a3a5c;}")
            self.caption.setStyleSheet("font-size: 10px; color: #5dade2; font-weight: bold;")
        else:
            self.img_label.setPixmap(self._gray_pix)
            self.setStyleSheet(
                "QFrame{border: 3px solid #444; border-radius: 4px; "
                "background: #1e1e1e;}")
            self.caption.setStyleSheet("font-size: 10px; color: #888;")

    def mousePressEvent(self, event):
        # 鼠标事件交由父容器统一处理（单击切换 + 预览）
        event.ignore()


# ===================== 单击网格容器 =====================

class GridContainer(QWidget):
    """单击缩略图在预览区显示（勾选框负责选中，点击本体不再切换选中）"""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            cell = self.childAt(event.position().toPoint())
            if isinstance(cell, QCheckBox):    # 点击选择框：交给复选框自身
                super().mousePressEvent(event)
                return
            # 向上查找 ThumbCell
            while cell is not None and not isinstance(cell, ThumbCell):
                cell = cell.parentWidget()
            if isinstance(cell, ThumbCell):
                self.page._preview_frame(cell.index)   # 单击仅预览
                event.accept()
                return
            event.ignore()
        else:
            super().mousePressEvent(event)


# ===================== 页面：帧选择 =====================

class FrameSelectorPage(QWidget):
    MODE_LOOP = 0
    MODE_PINGPONG = 1
    MODE_ONCE = 2

    def __init__(self, shared_state: dict = None, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state if shared_state is not None else {}
        self.frames: List[FrameItem] = []
        self.cells: List[ThumbCell] = []
        self.frames_dir: str = ""
        self._frames_sig: str = ""   # 帧目录内容签名（换视频重抽时变化）

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._next_play_frame)
        self.play_index = 0
        self.play_direction = 1
        self._current_cell_idx = -1  # 当前黄色高亮帧在 self.frames 中的索引

        # resize 防抖定时器，避免全屏切换时频繁重建网格
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._do_resize_rebuild)

        self._build_ui()

        # 键盘快捷键：A/D 切换上一张/下一张预览，空格 播放/暂停
        QShortcut(QKeySequence(Qt.Key_A), self, context=Qt.WindowShortcut,
                  activated=lambda: self._step_preview(-1))
        QShortcut(QKeySequence(Qt.Key_D), self, context=Qt.WindowShortcut,
                  activated=lambda: self._step_preview(1))
        QShortcut(QKeySequence(Qt.Key_Space), self, context=Qt.WindowShortcut,
                  activated=self._toggle_play)
        # Q/E 选中/取消选中当前预览帧
        QShortcut(QKeySequence(Qt.Key_Q), self, context=Qt.WindowShortcut,
                  activated=lambda: self._set_current_selected(True))
        QShortcut(QKeySequence(Qt.Key_E), self, context=Qt.WindowShortcut,
                  activated=lambda: self._set_current_selected(False))

    def showEvent(self, event):
        super().showEvent(event)
        # 切到此页面时，若有共享的帧目录：路径变化或同目录被重新抽帧都要重载
        d = self.shared_state.get("frames_dir", "")
        if d and os.path.isdir(d):
            sig = self._frames_signature(d)
            if d != self.frames_dir or sig != self._frames_sig:
                self.dir_edit.setText(d)
                self._load_frames(d)

    @staticmethod
    def _frames_signature(directory: str) -> str:
        """帧目录内容签名：优先 frames_info.json 修改时间，否则用文件名清单"""
        info_path = os.path.join(directory, "frames_info.json")
        if os.path.isfile(info_path):
            try:
                return f"info:{os.path.getmtime(info_path):.3f}"
            except OSError:
                pass
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        try:
            return "files:" + "|".join(
                sorted(f for f in os.listdir(directory)
                       if f.lower().endswith(exts)))
        except OSError:
            return ""

    def _build_ui(self):
        self.setStyleSheet("QGroupBox{border:none;margin-top:8px;} QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        # 主体：左右布局（固定黄金比例 13:8）
        self.h_layout = QHBoxLayout()
        self.h_layout.setSpacing(8)
        root.addLayout(self.h_layout, 1)

        # ===== 左侧：动画预览（仅画面） =====
        preview_box = QGroupBox("动画预览（仅播放选中帧）")
        pl = QVBoxLayout(preview_box)
        self.preview_label = QLabel("未加载帧")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet(
            "QLabel{background:#2b2b2b;color:#bdc3c7;border:1px solid #444;}")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pl.addWidget(self.preview_label, 1)

        self.h_layout.addWidget(preview_box, 13)

        # ===== 右侧：帧目录 + 播放控制 + 选择操作 + 帧网格 + 状态 =====
        right_panel = QWidget()
        rlay = QVBoxLayout(right_panel)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(8)

        # 帧目录
        dir_box = QGroupBox("帧目录")
        dl = QHBoxLayout(dir_box)
        dl.addWidget(QLabel("目录:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("选择模块一输出的 frames 目录...")
        btn_load = QPushButton("加载")
        btn_load.clicked.connect(self._browse_dir)
        dl.addWidget(self.dir_edit, 1)
        dl.addWidget(btn_load)
        rlay.addWidget(dir_box, 0)

        # 播放控制区
        play_box = QGroupBox("播放控制")
        playl = QVBoxLayout(play_box)

        ctrl = QHBoxLayout()
        btn_prev = QPushButton("⏮ 上一帧")
        btn_prev.clicked.connect(self._prev_frame)
        ctrl.addWidget(btn_prev)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setMinimumHeight(32)
        ctrl.addWidget(self.btn_play)

        btn_next = QPushButton("⏭ 下一帧")
        btn_next.clicked.connect(self._next_frame)
        ctrl.addWidget(btn_next)

        ctrl.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(12)
        self.fps_spin.valueChanged.connect(self._update_timer)
        ctrl.addWidget(self.fps_spin)
        playl.addLayout(ctrl)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("模式:"))
        self.mode_group = QButtonGroup(self)
        self.rb_loop = QRadioButton("循环")
        self.rb_pingpong = QRadioButton("来回")
        self.rb_once = QRadioButton("一次性")
        self.rb_loop.setChecked(True)
        self.mode_group.addButton(self.rb_loop, self.MODE_LOOP)
        self.mode_group.addButton(self.rb_pingpong, self.MODE_PINGPONG)
        self.mode_group.addButton(self.rb_once, self.MODE_ONCE)
        mode_row.addWidget(self.rb_loop)
        mode_row.addWidget(self.rb_pingpong)
        mode_row.addWidget(self.rb_once)
        mode_row.addStretch()
        self.frame_pos_label = QLabel("- / -")
        self.frame_pos_label.setStyleSheet("color:#666;")
        mode_row.addWidget(self.frame_pos_label)
        playl.addLayout(mode_row)

        rlay.addWidget(play_box, 0)

        # 选择操作区
        func_box = QGroupBox("选择操作")
        fl = QVBoxLayout(func_box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("每 N 帧预选:"))
        self.preselect_spin = QSpinBox()
        self.preselect_spin.setRange(1, 10000)
        self.preselect_spin.setValue(2)
        row1.addWidget(self.preselect_spin)
        btn_preselect = QPushButton("应用")
        btn_preselect.clicked.connect(self._preselect)
        row1.addWidget(btn_preselect)
        fl.addLayout(row1)

        btn_row = QHBoxLayout()
        btn_sel_all = QPushButton("全选")
        btn_sel_all.clicked.connect(lambda: self._set_all(True))
        btn_row.addWidget(btn_sel_all)

        btn_sel_none = QPushButton("全不选")
        btn_sel_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(btn_sel_none)

        btn_invert = QPushButton("反选")
        btn_invert.clicked.connect(self._invert)
        btn_row.addWidget(btn_invert)
        fl.addLayout(btn_row)

        fl.addSpacing(8)
        self.count_label = QLabel("已选 0 / 0 帧")
        self.count_label.setStyleSheet("font-weight:bold; color:#2d7ff9; font-size:14px;")
        self.count_label.setAlignment(Qt.AlignCenter)
        fl.addWidget(self.count_label)

        btn_export = QPushButton("✓ 导出选中帧列表")
        btn_export.setMinimumHeight(34)
        btn_export.setStyleSheet(
            "QPushButton{background:#2d7ff9;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#1a6fe0;}")
        btn_export.clicked.connect(self._export_selection)
        fl.addWidget(btn_export)

        rlay.addWidget(func_box, 0)

        # 帧选择网格
        grid_box = QGroupBox("所有帧（左上角勾选=选中  |  单击=预览  |  A/D=上一张/下一张  |  Q/E=选中/取消选中  |  空格=播放/暂停  |  黄框=当前预览）")
        gl = QVBoxLayout(grid_box)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea{background:#2b2b2b;border:none;}")
        self.container = GridContainer(self)
        self.container.setStyleSheet("QWidget{background:#2b2b2b;}")
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(8)
        self.scroll.setWidget(self.container)
        gl.addWidget(self.scroll, 1)
        rlay.addWidget(grid_box, 1)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #555;")
        rlay.addWidget(self.status_label)

        # 右侧面板包裹滚动区，高缩放下内容可滚动
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("QScrollArea{background:#2b2b2b;border:none;}")
        right_scroll.setWidget(right_panel)
        self.h_layout.addWidget(right_scroll, 8)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择帧目录")
        if d:
            self.dir_edit.setText(d)
            self._load_frames(d)

    def _load_frames(self, directory: str):
        self.frames_dir = directory
        self._frames_sig = self._frames_signature(directory)
        # 手动加载的目录也同步给后续模块（抠图/色调/导出）
        self.shared_state["frames_dir"] = os.path.abspath(directory)
        self.frames = []
        self.cells = []
        self._stop_play()

        info_path = os.path.join(directory, "frames_info.json")
        info_map = {}
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for fr in data.get("frames", []):
                    if fr.get("is_duplicate"):
                        continue
                    info_map[fr["filename"]] = fr
            except Exception:
                pass

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        files = sorted([f for f in os.listdir(directory)
                        if f.lower().endswith(exts)])

        for i, fn in enumerate(files):
            fp = os.path.join(directory, fn)
            info = info_map.get(fn, {})
            item = FrameItem(
                index=i, filename=fn, filepath=fp,
                original_index=info.get("original_index", -1),
                timestamp_ms=info.get("timestamp_ms", 0.0),
            )
            self.frames.append(item)

        self._build_grid()
        self._update_count()
        self.status_label.setText(f"已加载 {len(self.frames)} 帧")

        sel_path = os.path.join(directory, "selected_frames.json")
        if os.path.isfile(sel_path):
            try:
                with open(sel_path, "r", encoding="utf-8") as f:
                    sel = json.load(f)
                indices = set(sel.get("selected_indices", []))
                for idx in indices:
                    if 0 <= idx < len(self.frames):
                        self.frames[idx].selected = True
                        self.cells[idx].set_selected(True)
                self._update_count()
            except Exception:
                pass

    def _build_grid(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.cells = []
        self._current_cell_idx = -1
        thumb_w = 110
        thumb_h = 80
        cell_w = thumb_w + 14 + 8  # 缩略图宽 + padding + spacing
        avail = self.scroll.viewport().width()
        cols = max(1, avail // cell_w) if avail > 0 else 4
        for item in self.frames:
            cell = ThumbCell(item.index, item.filepath, thumb_w, thumb_h)
            cell.check_toggled.connect(self._on_cell_check)
            self.cells.append(cell)
            r, c = divmod(item.index, cols)
            self.grid.addWidget(cell, r, c)
        self.grid.setRowStretch(self.grid.rowCount(), 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 防抖：resize 停止 150ms 后才重建网格
        if self.frames:
            self._resize_timer.start()

    def _do_resize_rebuild(self):
        if not self.frames:
            return
        # 记录当前播放帧，重建后恢复高亮
        sel = self._selected_frames()
        restore_idx = -1
        if sel and 0 <= self.play_index < len(sel):
            restore_idx = sel[self.play_index].index
        self._build_grid()
        for i, cell in enumerate(self.cells):
            if self.frames[i].selected:
                cell.set_selected(True)
        if restore_idx >= 0 and restore_idx < len(self.cells):
            self._set_current_cell(restore_idx)

    def _on_cell_check(self, index: int, checked: bool):
        """左上角选择框勾选变化 → 更新选中状态"""
        if 0 <= index < len(self.frames):
            self.frames[index].selected = checked
            self.cells[index].set_selected(checked)
            self._update_count()

    def _set_current_selected(self, sel: bool):
        """Q/E：选中/取消选中当前预览帧（黄框所在帧）"""
        idx = self._current_cell_idx
        if 0 <= idx < len(self.frames):
            self._on_cell_check(idx, sel)

    def _set_all(self, value: bool):
        for f in self.frames:
            f.selected = value
        for cell in self.cells:
            cell.set_selected(value)
        self._update_count()

    def _invert(self):
        for f in self.frames:
            f.selected = not f.selected
        for i, cell in enumerate(self.cells):
            cell.set_selected(self.frames[i].selected)
        self._update_count()

    def _preselect(self):
        n = self.preselect_spin.value()
        if n <= 0:
            return
        for f in self.frames:
            f.selected = (f.index % n == 0)
        for i, cell in enumerate(self.cells):
            cell.set_selected(self.frames[i].selected)
        self._update_count()

    def _update_count(self):
        sel = sum(1 for f in self.frames if f.selected)
        total = len(self.frames)
        self.count_label.setText(f"已选 {sel} / {total} 帧")
        # 实时同步选中帧给后续模块（抠图/色调/导出），无需等待导出按钮
        sel_frames = [f for f in self.frames if f.selected]
        self.shared_state["selected_indices"] = [f.index for f in sel_frames]
        self.shared_state["selected_files"] = [f.filename for f in sel_frames]

    def _selected_frames(self) -> List[FrameItem]:
        return [f for f in self.frames if f.selected]

    def _toggle_play(self):
        if self.play_timer.isActive():
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        sel = self._selected_frames()
        if not sel:
            QMessageBox.information(self, "提示", "请先选择至少一帧！")
            return
        self.play_index = 0
        self.play_direction = 1
        self._update_timer()
        self.play_timer.start()
        self.btn_play.setText("⏸ 暂停")
        self._show_play_frame()

    def _stop_play(self):
        self.play_timer.stop()
        self.btn_play.setText("▶ 播放")

    def _prev_frame(self):
        """手动切换到上一帧"""
        sel = self._selected_frames()
        if not sel:
            return
        n = len(sel)
        self.play_index = (self.play_index - 1) % n
        self._show_play_frame()

    def _next_frame(self):
        """手动切换到下一帧"""
        sel = self._selected_frames()
        if not sel:
            return
        n = len(sel)
        self.play_index = (self.play_index + 1) % n
        self._show_play_frame()

    def _update_timer(self):
        fps = self.fps_spin.value()
        interval = int(1000 / max(0.1, fps))
        self.play_timer.setInterval(interval)

    def _next_play_frame(self):
        sel = self._selected_frames()
        if not sel:
            self._stop_play()
            return
        n = len(sel)
        mode = self.mode_group.checkedId()

        if mode == self.MODE_LOOP:
            self.play_index = (self.play_index + 1) % n
        elif mode == self.MODE_PINGPONG:
            self.play_index += self.play_direction
            if self.play_index >= n - 1:
                self.play_index = n - 1
                self.play_direction = -1
            elif self.play_index <= 0:
                self.play_index = 0
                self.play_direction = 1
        else:
            self.play_index += 1
            if self.play_index >= n:
                self._stop_play()
                return
        self._show_play_frame()

    def _show_play_frame(self):
        sel = self._selected_frames()
        if not sel or self.play_index >= len(sel):
            return
        item = sel[self.play_index]
        pix = QPixmap(item.filepath)
        if not pix.isNull():
            scaled = pix.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
        self._set_current_cell(item.index)
        self._update_pos_label()

    def _set_current_cell(self, idx: int):
        """在网格中用黄色边框标记当前播放的帧"""
        if idx == self._current_cell_idx:
            return
        # 清除上一个高亮
        if 0 <= self._current_cell_idx < len(self.cells):
            self.cells[self._current_cell_idx].set_current(False)
        # 设置新高亮
        if 0 <= idx < len(self.cells):
            self.cells[idx].set_current(True)
        self._current_cell_idx = idx

    def _step_preview(self, delta: int):
        """A/D：在全部帧中切换上一张/下一张预览（黄色高亮跟随）"""
        n = len(self.frames)
        if n == 0:
            return
        cur = self._current_cell_idx
        if 0 <= cur < n:
            idx = (cur + delta) % n
        else:
            idx = 0 if delta > 0 else n - 1
        self._preview_frame(idx)

    def _preview_frame(self, idx: int):
        """在预览区显示指定索引的帧（用于单击缩略图查看）"""
        if not (0 <= idx < len(self.frames)):
            return
        item = self.frames[idx]
        pix = QPixmap(item.filepath)
        if not pix.isNull():
            scaled = pix.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
        self._set_current_cell(idx)
        # 若该帧已选中，同步播放索引，便于上一帧/下一帧继续
        sel = self._selected_frames()
        for i, f in enumerate(sel):
            if f.index == idx:
                self.play_index = i
                self._update_pos_label()
                break
        else:
            self.frame_pos_label.setText(
                f"预览 原帧 #{idx + 1}（未在选中列表中）")

    def _update_pos_label(self):
        sel = self._selected_frames()
        if sel:
            idx = min(self.play_index, len(sel) - 1)
            self.frame_pos_label.setText(
                f"{idx + 1} / {len(sel)}  (原帧 #{sel[idx].index + 1})")

    def _export_selection(self):
        sel = self._selected_frames()
        if not sel:
            QMessageBox.warning(self, "提示", "没有选中的帧！")
            return
        if not self.frames_dir:
            QMessageBox.warning(self, "提示", "未加载帧目录！")
            return

        out = {
            "frames_dir": os.path.abspath(self.frames_dir),
            "total_frames": len(self.frames),
            "selected_count": len(sel),
            "selected_indices": [f.index for f in sel],
            "selected_files": [f.filename for f in sel],
            "selected_original_indices": [f.original_index for f in sel],
            "selected_timestamps_ms": [f.timestamp_ms for f in sel],
        }
        out_path = os.path.join(self.frames_dir, "selected_frames.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # 共享选中帧给后续页面
        self.shared_state["selected_frames"] = [f.filename for f in sel]
        self.shared_state["selected_indices"] = [f.index for f in sel]

        self.status_label.setText(f"已导出选中帧列表: {out_path}")
        QMessageBox.information(self, "导出成功", f"已导出 {len(sel)} 帧到:\n{out_path}")

    def closeEvent(self, event):
        self._stop_play()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = QMainWindow()
    win.setWindowTitle("帧二次筛选 + 动画预览")
    win.resize(1200, 820)
    page = FrameSelectorPage()
    win.setCentralWidget(page)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
