# -*- coding: utf-8 -*-
"""
模块七：导出图片 - 页面

将上游制作好的序列帧拼接为一张精灵表（sprite sheet）：
  - 来源优先级：调整位置 adjusted → 色调统一 colored → 精抠 refined → 抠图 matted
  - 默认一行排列，可切自动近方形或手动列数，间距可调
  - 输出分辨率按目标宽高像素设置，可约束比例联动
  - 导出到用户指定文件夹，文件名可命名（默认 sprite_sheet.png）
  - 动画预览：重复 / 来回两种播放方式，FPS 可调
"""

import os
import json
import math
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QSpinBox, QCheckBox, QRadioButton, QButtonGroup,
    QLineEdit, QFileDialog,
)

from gui_frame_matting import (imread_unicode, imwrite_unicode,
                               compose_checkerboard, frame_no_from_name)
from gui_frame_adjust import bgra_to_pixmap


INFO_FILE = "export_info.json"
DEFAULT_NAME = "sprite_sheet"


def compose_sheet(images: List[np.ndarray], cols: int, spacing: int
                  ) -> np.ndarray:
    """把 BGRA 帧按网格拼成一张精灵表（每格居中，透明底）"""
    n = len(images)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    cw = max(im.shape[1] for im in images)
    ch = max(im.shape[0] for im in images)
    W = cols * cw + (cols - 1) * spacing
    H = rows * ch + (rows - 1) * spacing
    sheet = np.zeros((H, W, 4), np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x0 = c * (cw + spacing) + (cw - im.shape[1]) // 2
        y0 = r * (ch + spacing) + (ch - im.shape[0]) // 2
        sub = sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]].astype(np.float32)
        sa = im[:, :, 3:4].astype(np.float32) / 255.0
        da = sub[:, :, 3:4] / 255.0
        ao = sa + da * (1.0 - sa)
        prem = im[:, :, :3].astype(np.float32) * sa + \
            sub[:, :, :3] * da * (1.0 - sa)
        safe = np.clip(ao, 1e-6, None)
        out = np.empty_like(sub)
        out[:, :, :3] = np.where(ao > 1e-6, prem / safe, 0.0)
        out[:, :, 3:4] = ao * 255.0
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = \
            np.clip(out, 0, 255).astype(np.uint8)
    return sheet


def auto_cols(n: int, cell_w: int, cell_h: int) -> int:
    """近方形自动列数（按像素宽高比）"""
    if n <= 1 or cell_w <= 0 or cell_h <= 0:
        return max(1, n)
    return max(1, min(n, int(round(math.sqrt(n * cell_h / cell_w)))))


def resize_sheet(sheet: np.ndarray, w: int, h: int) -> np.ndarray:
    """缩放到目标宽高（像素），100% 原样返回"""
    w, h = max(1, int(w)), max(1, int(h))
    if w == sheet.shape[1] and h == sheet.shape[0]:
        return sheet
    interp = cv2.INTER_AREA if w * h < sheet.shape[0] * sheet.shape[1] \
        else cv2.INTER_LANCZOS4
    return cv2.resize(sheet, (w, h), interpolation=interp)


class FrameExportPage(QWidget):
    """导出图片页：精灵表拼接 + 分辨率调整 + 动画预览 + 导出"""

    def __init__(self, shared_state: dict = None, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state if shared_state is not None else {}
        self.frames: List[Tuple[str, int]] = []   # (path, 帧号)
        self.src_dir: str = ""
        self._sig: Optional[str] = None   # None=从未加载，保证首次show必重载
        self._images: List[Optional[np.ndarray]] = []
        self._sheet_cache = None                  # (sig_params, BGRA 100%)
        self._pm_cache: Dict[int, QPixmap] = {}   # 动画预览帧缓存
        self.play_index = 0
        self.play_dir = 1
        self._link_guard = False   # 约束比例联动时防止信号递归

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._rebuild_sheet_preview)

        self._build_ui()

    # ---------------- 来源 ----------------

    def _source_dir(self) -> str:
        for key in ("adjusted_dir", "colored_dir", "refined_dir",
                    "matted_dir"):
            d = self.shared_state.get(key, "")
            if d and os.path.isdir(d):
                return d
        return ""

    @staticmethod
    def _dir_sig(d: str) -> str:
        try:
            files = [f for f in os.listdir(d) if f.lower().endswith(".png")
                     and not f.startswith("_")]
            if not files:
                return ""
            newest = max(os.path.getmtime(os.path.join(d, f))
                         for f in files)
            return f"{d}|{len(files)}|{newest:.3f}"
        except OSError:
            return ""

    def showEvent(self, event):
        super().showEvent(event)
        d = self._source_dir()
        sig = self._dir_sig(d) if d else ""
        if sig != self._sig:
            self._reload(d, sig)

    def _reload(self, d: str, sig: str):
        self.play_timer.stop()
        self.play_index = 0
        self.play_dir = 1
        self._pm_cache.clear()
        self._sheet_cache = None
        self._sig = sig
        self.src_dir = d
        self.frames = []
        self._images = []
        if d:
            names = [f for f in os.listdir(d)
                     if f.lower().endswith(".png")
                     and not f.startswith("_")]
            names.sort(key=lambda f: frame_no_from_name(f, 0))
            self.frames = [(os.path.join(d, f),
                            frame_no_from_name(f, i + 1))
                           for i, f in enumerate(names)]
            self._images = [None] * len(self.frames)
        src_name = {"adjusted_dir": "调整位置", "colored_dir": "色调统一",
                    "refined_dir": "图片精抠", "matted_dir": "批量抠图"}
        kind = next((v for k, v in src_name.items()
                     if self.shared_state.get(k, "") == d), "")
        n = len(self.frames)
        self.src_label.setText(
            f"来源：{kind}（{os.path.basename(d)}）  {n} 帧" if n
            else "未找到上游图片（请先完成抠图/精抠/色调统一/调整位置）")
        has = n > 0
        self.btn_play.setEnabled(has)
        self.btn_export.setEnabled(has)
        self._update_play_btn()
        if not self._load_prefs():
            # 无偏好记录：默认一行排列，输出分辨率=原图尺寸
            self.rows_spin.setValue(1)
            dims = self._orig_dims()
            if dims:
                self._set_out_size(*dims)
        self._update_size_label()
        self._show_play_frame()
        self._debounce.start()

    def _image(self, i: int) -> Optional[np.ndarray]:
        if self._images[i] is None:
            img = imread_unicode(self.frames[i][0])
            if img is not None and img.ndim == 3 and img.shape[2] == 3:
                a = np.full(img.shape[:2] + (1,), 255, np.uint8)
                img = np.concatenate([img, a], axis=2)
            self._images[i] = img
        return self._images[i]

    # ---------------- UI ----------------

    def _build_ui(self):
        self.setStyleSheet(
            "QGroupBox{border:none;margin-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;"
            "padding:0 4px;}")
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ===== 左侧：动画预览 + 精灵表预览 =====
        left = QVBoxLayout()
        left.setSpacing(8)

        anim_box = QGroupBox("动画预览（按当前帧序列播放）")
        av = QVBoxLayout(anim_box)
        self.anim_label = QLabel("未加载帧")
        self.anim_label.setAlignment(Qt.AlignCenter)
        self.anim_label.setMinimumSize(400, 260)
        self.anim_label.setStyleSheet("QLabel{background:#2b2b2b;}")
        av.addWidget(self.anim_label, 1)
        rowp = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self._toggle_play)
        rowp.addWidget(self.btn_play)
        rowp.addWidget(QLabel(" 播放方式："))
        self.rb_loop = QRadioButton("重复")
        self.rb_ping = QRadioButton("来回")
        self.rb_loop.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_loop)
        bg.addButton(self.rb_ping)
        rowp.addWidget(self.rb_loop)
        rowp.addWidget(self.rb_ping)
        rowp.addWidget(QLabel(" FPS："))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(12)
        self.fps_spin.valueChanged.connect(self._on_fps)
        rowp.addWidget(self.fps_spin)
        self.frame_lab = QLabel("0/0")
        rowp.addWidget(self.frame_lab)
        rowp.addStretch()
        av.addLayout(rowp)
        left.addWidget(anim_box, 3)

        sheet_box = QGroupBox("精灵表预览")
        sv = QVBoxLayout(sheet_box)
        self.sheet_label = QLabel("—")
        self.sheet_label.setAlignment(Qt.AlignCenter)
        self.sheet_label.setMinimumSize(400, 180)
        self.sheet_label.setStyleSheet("QLabel{background:#2b2b2b;}")
        sv.addWidget(self.sheet_label, 1)
        self.sheet_info = QLabel("")
        self.sheet_info.setStyleSheet("color:#888;font-size:12px;")
        self.sheet_info.setAlignment(Qt.AlignCenter)
        sv.addWidget(self.sheet_info)
        left.addWidget(sheet_box, 2)
        root.addLayout(left, 13)

        # ===== 右侧：参数 + 导出 =====
        right = QVBoxLayout()
        right.setSpacing(8)

        src_box = QGroupBox("图片信息")
        sf = QVBoxLayout(src_box)
        self.src_label = QLabel("未加载")
        self.src_label.setWordWrap(True)
        sf.addWidget(self.src_label)
        right.addWidget(src_box)

        param_box = QGroupBox("精灵图参数")
        pf = QVBoxLayout(param_box)
        row1 = QHBoxLayout()
        self.chk_auto = QCheckBox("自动网格（近方形）")
        self.chk_auto.setChecked(False)          # 默认一行排列
        self.chk_auto.toggled.connect(self._on_params)
        row1.addWidget(self.chk_auto)
        row1.addWidget(QLabel(" 行数："))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 1000)
        self.rows_spin.setValue(1)
        self.rows_spin.setEnabled(True)
        self.rows_spin.valueChanged.connect(self._on_params)
        row1.addWidget(self.rows_spin)
        row1.addStretch()
        pf.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("间距："))
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(0, 64)
        self.spacing_spin.setSuffix(" px")
        self.spacing_spin.setValue(0)
        self.spacing_spin.valueChanged.connect(self._on_params)
        row2.addWidget(self.spacing_spin)
        row2.addStretch()
        pf.addLayout(row2)
        row2b = QHBoxLayout()
        row2b.addWidget(QLabel("输出分辨率："))
        self.out_w = QSpinBox()
        self.out_w.setRange(1, 99999)
        self.out_w.setSuffix(" px")
        self.out_w.valueChanged.connect(self._on_out_w)
        row2b.addWidget(self.out_w)
        row2b.addWidget(QLabel("×"))
        self.out_h = QSpinBox()
        self.out_h.setRange(1, 99999)
        self.out_h.setSuffix(" px")
        self.out_h.valueChanged.connect(self._on_out_h)
        row2b.addWidget(self.out_h)
        self.chk_aspect = QCheckBox("约束比例")
        self.chk_aspect.setChecked(True)
        self.chk_aspect.setToolTip("勾选后宽/高按原图比例联动")
        self.chk_aspect.toggled.connect(self._on_aspect_toggled)
        row2b.addWidget(self.chk_aspect)
        row2b.addStretch()
        pf.addLayout(row2b)
        self.size_label = QLabel("输出尺寸：—")
        self.size_label.setStyleSheet("color:#5dade2;")
        pf.addWidget(self.size_label)
        right.addWidget(param_box)

        out_box = QGroupBox("导出")
        of = QVBoxLayout(out_box)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("文件夹："))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("选择导出文件夹…")
        row3.addWidget(self.out_edit, 1)
        btn_dir = QPushButton("浏览…")
        btn_dir.clicked.connect(self._browse_dir)
        row3.addWidget(btn_dir)
        of.addLayout(row3)
        row3b = QHBoxLayout()
        row3b.addWidget(QLabel("文件名："))
        self.name_edit = QLineEdit(DEFAULT_NAME)
        self.name_edit.setPlaceholderText(DEFAULT_NAME)
        row3b.addWidget(self.name_edit, 1)
        row3b.addWidget(QLabel(".png"))
        of.addLayout(row3b)
        self.btn_export = QPushButton("📤 导出精灵表 PNG")
        self.btn_export.setStyleSheet(
            "QPushButton{background:#2d7ff9;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:hover{background:#4a9bff;}"
            "QPushButton:disabled{background:#3a3a3a;color:#888;}")
        self.btn_export.clicked.connect(self._export)
        of.addWidget(self.btn_export)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#2ecc71;font-size:12px;")
        of.addWidget(self.status)
        right.addWidget(out_box)
        right.addStretch()
        root.addLayout(right, 8)

    # ---------------- 动画预览 ----------------

    def _on_fps(self, fps: int):
        if self.play_timer.isActive():
            self.play_timer.start(max(10, int(1000 / max(1, fps))))

    def _toggle_play(self):
        if not self.frames:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
        else:
            self.play_timer.start(max(10, int(1000 / self.fps_spin.value())))
        self._update_play_btn()

    def _update_play_btn(self):
        self.btn_play.setText("⏸ 暂停" if self.play_timer.isActive()
                              else "▶ 播放")

    def _advance(self):
        n = len(self.frames)
        if n == 0:
            self.play_timer.stop()
            self._update_play_btn()
            return
        if self.rb_loop.isChecked():               # 重复
            self.play_index = (self.play_index + 1) % n
        else:                                      # 来回
            nxt = self.play_index + self.play_dir
            if nxt >= n or nxt < 0:
                self.play_dir *= -1
                nxt = self.play_index + self.play_dir
                nxt = max(0, min(n - 1, nxt))
            self.play_index = nxt
        self._show_play_frame()

    def _show_play_frame(self):
        n = len(self.frames)
        self.frame_lab.setText(
            f"{self.play_index + 1}/{n}" if n else "0/0")
        if not n:
            self.anim_label.setText("未加载帧")
            return
        pm = self._pm_cache.get(self.play_index)
        if pm is None:
            img = self._image(self.play_index)
            if img is None:
                return
            pm = bgra_to_pixmap(img)
            if pm is None or pm.isNull():
                return
            self._pm_cache[self.play_index] = pm
        area = self.anim_label.size()
        pm2 = pm.scaled(area, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.anim_label.setPixmap(compose_checkerboard(pm2))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._pm_cache.clear()
        self._show_play_frame()
        self._debounce.start()

    # ---------------- 精灵表 ----------------

    def _grid_params(self) -> Tuple[int, int]:
        """返回 (列数, 间距)；手动模式由行数推算列数"""
        spacing = self.spacing_spin.value()
        n = len(self.frames)
        if self.chk_auto.isChecked():
            imgs = [im for im in (self._image(i)
                                  for i in range(n))
                    if im is not None]
            if not imgs:
                return 1, spacing
            cw = max(im.shape[1] for im in imgs)
            ch = max(im.shape[0] for im in imgs)
            return auto_cols(len(imgs), cw, ch), spacing
        rows = max(1, self.rows_spin.value())
        cols = (n + rows - 1) // rows if n else 1
        return max(1, cols), spacing

    def _orig_dims(self) -> Optional[Tuple[int, int]]:
        """100% 精灵表尺寸 (W, H)"""
        imgs = [im for im in (self._image(i)
                              for i in range(len(self.frames)))
                if im is not None]
        if not imgs:
            return None
        cw = max(im.shape[1] for im in imgs)
        ch = max(im.shape[0] for im in imgs)
        cols, spacing = self._grid_params()
        rows = (len(imgs) + cols - 1) // cols
        return (cols * cw + (cols - 1) * spacing,
                rows * ch + (rows - 1) * spacing)

    def _set_out_size(self, w: int, h: int):
        """程序化设置输出分辨率（不触发联动）"""
        self._link_guard = True
        self.out_w.setValue(max(1, w))
        self.out_h.setValue(max(1, h))
        self._link_guard = False

    def _relink(self, source: str):
        """约束比例联动：source='w' 时由宽算高，'h' 时由高算宽"""
        if not self.chk_aspect.isChecked() or self._link_guard:
            return
        dims = self._orig_dims()
        if not dims:
            return
        W0, H0 = dims
        self._link_guard = True
        if source == "w":
            self.out_h.setValue(
                max(1, int(self.out_w.value() * H0 / W0 + 0.5)))
        else:
            self.out_w.setValue(
                max(1, int(self.out_h.value() * W0 / H0 + 0.5)))
        self._link_guard = False

    def _on_out_w(self, _v):
        self._relink("w")
        self._update_size_label()
        self._debounce.start()

    def _on_out_h(self, _v):
        self._relink("h")
        self._update_size_label()
        self._debounce.start()

    def _on_aspect_toggled(self, on: bool):
        if on:
            self._relink("w")          # 重新勾选用当前宽回正高度
        self._update_size_label()
        self._debounce.start()

    def _sheet_100(self) -> Optional[np.ndarray]:
        """100% 尺寸精灵表（按参数缓存）"""
        imgs = [im for im in (self._image(i)
                              for i in range(len(self.frames)))
                if im is not None]
        if not imgs:
            return None
        cols, spacing = self._grid_params()
        sig = (cols, spacing, len(imgs), self._sig)
        if self._sheet_cache and self._sheet_cache[0] == sig:
            return self._sheet_cache[1]
        sheet = compose_sheet(imgs, cols, spacing)
        self._sheet_cache = (sig, sheet)
        return sheet

    def _update_size_label(self):
        n = len(self.frames)
        if not n:
            self.size_label.setText("输出尺寸：—")
            return
        dims = self._orig_dims()
        if not dims:
            return
        W, H = dims
        cols, _ = self._grid_params()
        rows = (n + cols - 1) // cols
        self.size_label.setText(
            f"输出尺寸：{self.out_w.value()}×{self.out_h.value()}"
            f"（{cols}×{rows} 网格，原图 {W}×{H}）")

    def _on_params(self, *args):
        self.rows_spin.setEnabled(not self.chk_auto.isChecked())
        self._relink("w")      # 网格变化改了原图比例，用宽回正高度
        self._update_size_label()
        self._debounce.start()

    def _rebuild_sheet_preview(self):
        sheet = self._sheet_100()
        if sheet is None:
            self.sheet_label.setText("—")
            self.sheet_info.setText("")
            return
        out = resize_sheet(sheet, self.out_w.value(), self.out_h.value())
        h, w = out.shape[:2]
        qpm = bgra_to_pixmap(out)
        if qpm is None or qpm.isNull():
            return
        pm2 = qpm.scaled(self.sheet_label.size(), Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        self.sheet_label.setPixmap(compose_checkerboard(pm2))
        cols, _ = self._grid_params()
        rows = (len(self.frames) + cols - 1) // cols
        self.sheet_info.setText(f"{cols}×{rows} 网格，输出 {w}×{h}")

    # ---------------- 导出 ----------------

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出文件夹",
                                             self.out_edit.text() or "")
        if d:
            self.out_edit.setText(d)
            self._save_prefs()

    def _export(self):
        sheet = self._sheet_100()
        if sheet is None:
            self.status.setText("没有可导出的图片")
            return
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            out_dir = os.path.join(os.path.abspath("frames"), "export")
            self.out_edit.setText(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out = resize_sheet(sheet, self.out_w.value(), self.out_h.value())
        name = self.name_edit.text().strip() or DEFAULT_NAME
        if not name.lower().endswith(".png"):
            name += ".png"
        dst = os.path.join(out_dir, name)
        if not imwrite_unicode(dst, out):
            self.status.setText("导出失败：无法写入文件")
            return
        h, w = out.shape[:2]
        self.status.setText(f"已导出 → {dst}（{w}×{h}）")
        self._save_prefs()

    # ---------------- 偏好持久化 ----------------

    def _prefs_path(self) -> str:
        fd = self.shared_state.get("frames_dir", "")
        return os.path.join(fd, INFO_FILE) if fd else ""

    def _save_prefs(self):
        p = self._prefs_path()
        if not p:
            return
        data = {
            "auto_grid": self.chk_auto.isChecked(),
            "rows": self.rows_spin.value(),
            "spacing": self.spacing_spin.value(),
            "out_w": self.out_w.value(),
            "out_h": self.out_h.value(),
            "keep_aspect": self.chk_aspect.isChecked(),
            "name": self.name_edit.text(),
            "out_dir": self.out_edit.text(),
            "fps": self.fps_spin.value(),
            "play_mode": "loop" if self.rb_loop.isChecked() else "pingpong",
        }
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def _load_prefs(self) -> bool:
        """读偏好并回填；无记录返回 False（调用方给默认值）"""
        p = self._prefs_path()
        if not p or not os.path.isfile(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        self.chk_auto.setChecked(d.get("auto_grid", False))
        self.rows_spin.setValue(d.get("rows", 1))
        self.spacing_spin.setValue(d.get("spacing", 0))
        self.chk_aspect.setChecked(d.get("keep_aspect", True))
        dims = self._orig_dims()
        self._set_out_size(d.get("out_w", dims[0] if dims else 1),
                           d.get("out_h", dims[1] if dims else 1))
        self.name_edit.setText(d.get("name", DEFAULT_NAME))
        self.out_edit.setText(d.get("out_dir", ""))
        self.fps_spin.setValue(d.get("fps", 12))
        if d.get("play_mode") == "pingpong":
            self.rb_ping.setChecked(True)
        else:
            self.rb_loop.setChecked(True)
        self._on_params()
        return True
