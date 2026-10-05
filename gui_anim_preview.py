# -*- coding: utf-8 -*-
"""
模块八：动画预览 - 页面（独立工具，不参与会话存档）

用途：检查一个角色/物体的多个动画之间切换是否流畅。
  - 导入多张精灵表，每张切片成一个动画（列数/行数可调）
  - 每个动画独立的 FPS 与播放方式（循环/来回）
  - 点击列表切换动画，对比观察哪些动画需要改进
  - 「保存动画系列」把精灵表副本+配置写入用户指定文件夹（anim_series.json），
    「打开动画系列」随时重新载入
"""

import os
import json
import shutil
from typing import List, Optional, Dict

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QSpinBox, QRadioButton, QButtonGroup, QLineEdit, QFileDialog,
    QListWidget, QListWidgetItem, QScrollArea, QSizePolicy, QMessageBox,
)

from gui_frame_matting import imread_unicode, compose_checkerboard
from gui_frame_adjust import bgra_to_pixmap

SERIES_FILE = "anim_series.json"


def slice_sheet(sheet: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
    """把精灵表按 行×列 均切成帧序列（先行后列）"""
    rows = max(1, rows)
    cols = max(1, cols)
    h, w = sheet.shape[:2]
    ch, cw = h // rows, w // cols
    if ch <= 0 or cw <= 0:
        return []
    return [sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy()
            for r in range(rows) for c in range(cols)]


class AnimPreviewPage(QWidget):
    """动画预览页：多精灵表动画切换对比（数据独立保存，不入会话存档）"""

    def __init__(self, shared_state: dict = None, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state if shared_state is not None else {}
        self.anims: List[Dict] = []          # {name,sheet_path,fps,mode,rows,cols}
        self._sheets: Dict[int, np.ndarray] = {}   # 索引 -> BGRA 整表
        self._frames: List[np.ndarray] = []  # 当前动画切片
        self.current = -1
        self.play_index = 0
        self.play_dir = 1
        self._loading = False                # 回填控件时屏蔽信号

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance)

        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self):
        self.setStyleSheet(
            "QGroupBox{border:none;margin-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;"
            "padding:0 4px;}")
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ===== 左侧：预览 + 播放条 =====
        left = QVBoxLayout()
        left.setSpacing(8)
        preview_box = QGroupBox("动画预览")
        pv = QVBoxLayout(preview_box)
        self.anim_label = QLabel("尚未导入精灵表")
        self.anim_label.setAlignment(Qt.AlignCenter)
        self.anim_label.setMinimumSize(400, 300)
        self.anim_label.setStyleSheet("QLabel{background:#2b2b2b;}")
        self.anim_label.setSizePolicy(QSizePolicy.Expanding,
                                      QSizePolicy.Expanding)
        pv.addWidget(self.anim_label, 1)

        bar = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self._toggle_play)
        bar.addWidget(self.btn_play)
        btn_prev = QPushButton("⏮ 上一帧")
        btn_prev.clicked.connect(lambda: self._step(-1))
        bar.addWidget(btn_prev)
        btn_next = QPushButton("⏭ 下一帧")
        btn_next.clicked.connect(lambda: self._step(1))
        bar.addWidget(btn_next)
        self.frame_lab = QLabel("0/0")
        bar.addWidget(self.frame_lab)
        bar.addStretch()
        pv.addLayout(bar)
        left.addWidget(preview_box, 1)
        root.addLayout(left, 13)

        # ===== 右侧：动画列表 + 当前动画设置 + 动画系列 =====
        right_panel = QWidget()
        rlay = QVBoxLayout(right_panel)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(8)

        list_box = QGroupBox("动画列表（点击切换）")
        ll = QVBoxLayout(list_box)
        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self._on_select)
        ll.addWidget(self.listw, 1)
        brow = QHBoxLayout()
        btn_import = QPushButton("📥 导入精灵表…")
        btn_import.clicked.connect(self._import_sheets)
        brow.addWidget(btn_import, 1)
        self.btn_del = QPushButton("删除")
        self.btn_del.clicked.connect(self._delete_current)
        brow.addWidget(self.btn_del)
        ll.addLayout(brow)
        rlay.addWidget(list_box, 1)

        set_box = QGroupBox("当前动画设置")
        sl = QVBoxLayout(set_box)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("名称:"))
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name)
        nrow.addWidget(self.name_edit, 1)
        sl.addLayout(nrow)

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("列数:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 200)
        self.cols_spin.valueChanged.connect(self._on_grid)
        grid_row.addWidget(self.cols_spin)
        grid_row.addWidget(QLabel(" 行数:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 200)
        self.rows_spin.setValue(1)
        self.rows_spin.valueChanged.connect(self._on_grid)
        grid_row.addWidget(self.rows_spin)
        self.slice_lab = QLabel("")
        self.slice_lab.setStyleSheet("color:#888;")
        grid_row.addWidget(self.slice_lab, 1)
        sl.addLayout(grid_row)

        play_row = QHBoxLayout()
        play_row.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(12)
        self.fps_spin.valueChanged.connect(self._on_fps)
        play_row.addWidget(self.fps_spin)
        play_row.addWidget(QLabel(" 播放方式:"))
        self.rb_loop = QRadioButton("循环")
        self.rb_ping = QRadioButton("来回")
        self.rb_loop.setChecked(True)
        pbg = QButtonGroup(self)
        pbg.addButton(self.rb_loop)
        pbg.addButton(self.rb_ping)
        self.rb_loop.toggled.connect(self._on_mode)
        play_row.addWidget(self.rb_loop)
        play_row.addWidget(self.rb_ping)
        play_row.addStretch()
        sl.addLayout(play_row)
        rlay.addWidget(set_box, 0)

        series_box = QGroupBox("动画系列")
        ser = QHBoxLayout(series_box)
        btn_save = QPushButton("💾 保存动画系列…")
        btn_save.clicked.connect(self._save_series)
        ser.addWidget(btn_save, 1)
        btn_open = QPushButton("📂 打开动画系列…")
        btn_open.clicked.connect(self._open_series)
        ser.addWidget(btn_open, 1)
        rlay.addWidget(series_box, 0)

        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#555;")
        self.status.setWordWrap(True)
        rlay.addWidget(self.status)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet(
            "QScrollArea{background:#2b2b2b;border:none;}")
        right_scroll.setWidget(right_panel)
        root.addWidget(right_scroll, 8)

        self._refresh_enabled()

    def _refresh_enabled(self):
        has = self.current >= 0
        for w in (self.btn_play, self.btn_del, self.name_edit,
                  self.cols_spin, self.rows_spin, self.fps_spin,
                  self.rb_loop, self.rb_ping):
            w.setEnabled(has)

    # ---------------- 动画管理 ----------------

    def _import_sheets(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择精灵表图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)")
        if not paths:
            return
        for p in paths:
            sheet = imread_unicode(p)
            if sheet is None:
                continue
            if sheet.ndim == 3 and sheet.shape[2] == 3:
                a = np.full(sheet.shape[:2] + (1,), 255, np.uint8)
                sheet = np.concatenate([sheet, a], axis=2)
            name = os.path.splitext(os.path.basename(p))[0]
            # 重名加后缀
            names = {a["name"] for a in self.anims}
            base, i = name, 2
            while name in names:
                name = f"{base}_{i}"
                i += 1
            self.anims.append({"name": name, "sheet_path": os.path.abspath(p),
                               "fps": 12, "mode": "loop",
                               "rows": 1, "cols": 1})
            self._sheets[len(self.anims) - 1] = sheet
            self.listw.addItem(name)
        self.listw.setCurrentRow(len(self.anims) - 1)
        self.status.setText(f"已导入 {len(paths)} 张精灵表，共 {len(self.anims)} 个动画")

    def _delete_current(self):
        i = self.current
        if i < 0:
            return
        self.play_timer.stop()
        self.anims.pop(i)
        self._sheets = {((k - 1) if k > i else k): v
                        for k, v in self._sheets.items() if k != i}
        self.listw.takeItem(i)
        if not self.anims:
            self.current = -1
            self._frames = []
            self.anim_label.clear()
            self.anim_label.setText("尚未导入精灵表")
            self.frame_lab.setText("0/0")
            self._refresh_enabled()
        else:
            self.listw.setCurrentRow(min(i, len(self.anims) - 1))

    def _on_select(self, row: int):
        if self._loading or row < 0 or row >= len(self.anims):
            return
        was_playing = self.play_timer.isActive()
        self.play_timer.stop()
        self.current = row
        self.play_index = 0
        self.play_dir = 1
        a = self.anims[row]
        self._loading = True
        try:
            self.name_edit.setText(a["name"])
            self.rows_spin.setValue(a["rows"])
            self.cols_spin.setValue(a["cols"])
            self.fps_spin.setValue(a["fps"])
            (self.rb_loop if a["mode"] == "loop" else self.rb_ping
             ).setChecked(True)
        finally:
            self._loading = False
        self._reslice()
        self._refresh_enabled()
        if was_playing:
            self._start_timer()

    def _on_name(self):
        if self._loading or self.current < 0:
            return
        name = self.name_edit.text().strip()
        if not name:
            return
        names = {a["name"] for j, a in enumerate(self.anims)
                 if j != self.current}
        if name in names:
            self.status.setText("名称已存在，未修改")
            self.name_edit.setText(self.anims[self.current]["name"])
            return
        self.anims[self.current]["name"] = name
        self.listw.item(self.current).setText(name)

    def _on_grid(self):
        if self._loading or self.current < 0:
            return
        a = self.anims[self.current]
        a["rows"] = self.rows_spin.value()
        a["cols"] = self.cols_spin.value()
        self.play_index = 0
        self.play_dir = 1
        self._reslice()

    def _on_fps(self):
        if self._loading or self.current < 0:
            return
        self.anims[self.current]["fps"] = self.fps_spin.value()
        if self.play_timer.isActive():
            self._start_timer()

    def _on_mode(self):
        if self._loading or self.current < 0:
            return
        self.anims[self.current]["mode"] = (
            "loop" if self.rb_loop.isChecked() else "pingpong")

    # ---------------- 切片与播放 ----------------

    def _reslice(self):
        if self.current < 0:
            return
        a = self.anims[self.current]
        sheet = self._sheets.get(self.current)
        if sheet is None:
            sheet = imread_unicode(a["sheet_path"])
            if sheet is None:
                self.status.setText(f"精灵表读取失败: {a['sheet_path']}")
                return
            if sheet.ndim == 3 and sheet.shape[2] == 3:
                al = np.full(sheet.shape[:2] + (1,), 255, np.uint8)
                sheet = np.concatenate([sheet, al], axis=2)
            self._sheets[self.current] = sheet
        self._frames = slice_sheet(sheet, a["rows"], a["cols"])
        h, w = sheet.shape[:2]
        self.slice_lab.setText(
            f"{len(self._frames)} 帧  {w // a['cols']}×{h // a['rows']}")
        self.play_index = min(self.play_index,
                              max(0, len(self._frames) - 1))
        self._show_frame()

    def _show_frame(self):
        n = len(self._frames)
        if self.current < 0 or n == 0:
            self.frame_lab.setText("0/0")
            return
        pm = bgra_to_pixmap(self._frames[self.play_index])
        pm = pm.scaled(self.anim_label.size(), Qt.KeepAspectRatio,
                       Qt.SmoothTransformation)
        self.anim_label.setPixmap(compose_checkerboard(pm))
        self.frame_lab.setText(f"{self.play_index + 1}/{n}")

    def _start_timer(self):
        fps = self.anims[self.current]["fps"] if self.current >= 0 else 12
        self.play_timer.start(max(1, int(1000 / max(1, fps))))
        self.btn_play.setText("⏸ 暂停")

    def _toggle_play(self):
        if self.current < 0 or not self._frames:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶ 播放")
        else:
            self._start_timer()

    def _step(self, d: int):
        if self.current < 0 or not self._frames:
            return
        self.play_timer.stop()
        self.btn_play.setText("▶ 播放")
        self.play_index = (self.play_index + d) % len(self._frames)
        self._show_frame()

    def _advance(self):
        n = len(self._frames)
        if n == 0 or self.current < 0:
            return
        mode = self.anims[self.current]["mode"]
        if mode == "pingpong" and n > 1:
            nxt = self.play_index + self.play_dir
            if nxt >= n or nxt < 0:
                self.play_dir *= -1
                nxt = self.play_index + self.play_dir
            self.play_index = nxt
        else:
            self.play_index = (self.play_index + 1) % n
        self._show_frame()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._frames:
            self._show_frame()

    # ---------------- 动画系列存取（独立文件，不参与会话存档） ----------------

    def _save_series(self):
        if not self.anims:
            QMessageBox.information(self, "提示", "还没有动画可保存。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存动画系列（精灵表副本将放在同一文件夹）",
            SERIES_FILE, "动画系列 (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        d = os.path.dirname(os.path.abspath(path))
        metas = []
        used = set()
        for i, a in enumerate(self.anims):
            safe = "".join(c if c not in '\\/:*?"<>|' else "_"
                           for c in a["name"]) or f"anim_{i + 1}"
            fn, k = f"{safe}.png", 2
            while fn in used:
                fn = f"{safe}_{k}.png"
                k += 1
            used.add(fn)
            dst = os.path.join(d, fn)
            try:
                if os.path.abspath(a["sheet_path"]) != os.path.abspath(dst):
                    shutil.copy2(a["sheet_path"], dst)
            except OSError as e:
                self.status.setText(f"保存失败: {e}")
                return
            metas.append({"name": a["name"], "file": fn,
                          "fps": a["fps"], "mode": a["mode"],
                          "rows": a["rows"], "cols": a["cols"]})
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"animations": metas}, f,
                          ensure_ascii=False, indent=1)
        except OSError as e:
            self.status.setText(f"保存失败: {e}")
            return
        self.status.setText(f"动画系列已保存 → {path}（{len(metas)} 个动画）")

    def _open_series(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开动画系列", "",
            "动画系列 (*.json);;所有文件 (*.*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            QMessageBox.warning(self, "提示", "动画系列文件无法读取。")
            return
        base = os.path.dirname(os.path.abspath(path))
        self.play_timer.stop()
        self.btn_play.setText("▶ 播放")
        self.anims = []
        self._sheets = {}
        self.listw.clear()
        self.current = -1
        skipped = 0
        for m in data.get("animations", []):
            sp = m.get("file", "")
            sp = sp if os.path.isabs(sp) else os.path.join(base, sp)
            if not os.path.isfile(sp):
                skipped += 1
                continue
            self.anims.append({
                "name": m.get("name") or os.path.splitext(
                    os.path.basename(sp))[0],
                "sheet_path": os.path.abspath(sp),
                "fps": int(m.get("fps", 12)),
                "mode": "pingpong" if m.get("mode") == "pingpong" else "loop",
                "rows": max(1, int(m.get("rows", 1))),
                "cols": max(1, int(m.get("cols", 1)))})
            self.listw.addItem(self.anims[-1]["name"])
        if self.anims:
            self.listw.setCurrentRow(0)
        else:
            self._refresh_enabled()
        msg = f"已打开动画系列：{len(self.anims)} 个动画"
        if skipped:
            msg += f"（{skipped} 个精灵表文件缺失被跳过）"
        self.status.setText(msg)
