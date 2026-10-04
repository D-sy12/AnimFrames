# -*- coding: utf-8 -*-
"""
模块六：导出图片 - 页面（可嵌入主窗口）

两大功能：
  1. 动画精灵图（sprite sheet）—— 把全部制作好的图片拼在一张透明底
     PNG 上，支持自动/手动列数、间距、整体缩放（调分辨率），输出尺寸
     实时结算显示并带预览。
  2. PSD 分层图 —— 每张图片是一个独立图层（类似 Photoshop），每个图层
     可单独设置缩放比例（%），导出 8 位 RGBA 的 .psd，PS / Krita /
     Clip Studio / Photopea 可直接打开。

图片来源优先级：色调统一 colored/ → 精抠 refined/ → 批量抠图 matted/。
产物目录：frames_dir/export/（sprite_sheet.png / layers.psd）。

布局与其它页面一致：左右固定 13:8。左侧导出预览（棋盘格透明底）+
底部状态；右侧两个 Tab 分别放两套导出控件，其中 PSD 的每个图层一行。
"""

import os
import json
import shutil
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QScrollArea, QGroupBox, QCheckBox, QSpinBox,
    QTabWidget, QFrame, QMessageBox,
)

import export_utils as eu
from gui_frame_matting import imwrite_unicode
from gui_frame_color import FitImageLabel, bgra_to_pixmap


# ===================== 导出工作线程 =====================

class _ExportWorker(QThread):
    """精灵图 / PSD 全分辨率导出线程"""
    progress = Signal(str)
    finished_ok = Signal(str, str)   # (mode, path)
    failed = Signal(str)

    def __init__(self, mode: str, items: List[dict], params: dict,
                 out_path: str):
        super().__init__()
        self.mode = mode               # "sprite" / "psd"
        self.items = items
        self.params = params
        self.out_path = out_path

    def run(self):
        try:
            if self.mode == "sprite":
                self._run_sprite()
            else:
                self._run_psd()
        except Exception as e:
            self.failed.emit(str(e))

    def _load(self, item: dict) -> np.ndarray:
        img = eu.load_bgra(item["src"])
        if img is None:
            raise IOError(f"读取失败：{item['src']}")
        return img

    def _run_sprite(self):
        p = self.params
        images = []
        for i, it in enumerate(self.items, 1):
            images.append(self._load(it))
            self.progress.emit(f"读取图片 {i}/{len(self.items)}")
        self.progress.emit("正在拼接精灵图…")
        canvas, _ = eu.build_sprite_sheet(
            images, columns=0 if p["auto_columns"] else p["columns"],
            scale=p["sprite_scale"] / 100.0, padding=p["padding"])
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        if not imwrite_unicode(self.out_path, canvas):
            raise IOError(f"写入失败：{self.out_path}")
        self.progress.emit(
            f"精灵图已导出：{os.path.basename(self.out_path)} "
            f"{canvas.shape[1]}×{canvas.shape[0]}")
        self.finished_ok.emit("sprite", self.out_path)

    def _run_psd(self):
        p = self.params
        sizes = [(it["w"], it["h"]) for it in self.items]
        cw = max(w for w, h in sizes)
        ch = max(h for w, h in sizes)
        layers = []
        composite = np.zeros((ch, cw, 4), np.uint8)
        for i, it in enumerate(self.items, 1):
            img = self._load(it)
            pct = p["layer_scales"].get(it["key"], 100)
            sim = eu.scale_bgra(img, pct / 100.0)
            x, y = eu.centered_xy(cw, ch, sim.shape[1], sim.shape[0])
            name = os.path.splitext(it["key"])[0][:80]
            layers.append((name, sim, x, y))
            eu.flatten_over(composite, sim, x, y)
            self.progress.emit(f"准备图层 {i}/{len(self.items)}")
        self.progress.emit("正在写入 PSD（PackBits 压缩）…")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        eu.write_layered_psd(self.out_path, cw, ch, layers, composite)
        self.progress.emit(
            f"PSD 已导出：{os.path.basename(self.out_path)} "
            f"{cw}×{ch}，{len(layers)} 个图层")
        self.finished_ok.emit("psd", self.out_path)


class _PreviewLoader(QThread):
    """把所有帧解码并缩小缓存，避免预览每次读全分辨率"""
    loaded_one = Signal(str, object)
    done = Signal()

    def __init__(self, items: List[dict], max_side: int = 360):
        super().__init__()
        self.items = items
        self.max_side = max_side
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        for it in self.items:
            if self._stop:
                break
            img = eu.load_bgra(it["src"])
            if img is None:
                continue
            h, w = img.shape[:2]
            f = min(1.0, self.max_side / max(h, w))
            if f < 1.0:
                img = cv2.resize(img, (max(1, int(w * f)),
                                       max(1, int(h * f))),
                                 interpolation=cv2.INTER_AREA)
            self.loaded_one.emit(it["key"], img)
        self.done.emit()


# ===================== PSD 图层行 =====================

class LayerRow(QFrame):
    """PSD 图层列表中的一行：序号+名称+原始尺寸+独立缩放%"""
    clicked = Signal(str)
    scale_changed = Signal(str, int)

    def __init__(self, key: str, number: int, w: int, h: int, pct: int):
        super().__init__()
        self.key = key
        self._selected = False
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(6)
        self.name_lab = QLabel(f"#{number:03d}  {key}  ({w}×{h})")
        self.name_lab.setStyleSheet("background:transparent;")
        self.name_lab.setToolTip(key)
        lay.addWidget(self.name_lab, 1)
        self.spin = QSpinBox()
        self.spin.setRange(5, 400)
        self.spin.setValue(pct)
        self.spin.setSuffix(" %")
        self.spin.setFixedWidth(82)
        self.spin.setCursor(Qt.ArrowCursor)
        self.spin.valueChanged.connect(
            lambda v: self.scale_changed.emit(self.key, int(v)))
        lay.addWidget(self.spin)
        self._set_style()

    def set_selected(self, sel: bool):
        self._selected = sel
        self._set_style()

    def _set_style(self):
        if self._selected:
            border, bg, fg = "#f1c40f", "#1a3a5c", "#f1c40f"
        else:
            border, bg, fg = "#444444", "#1e1e1e", "#bbbbbb"
        self.setStyleSheet(
            f"LayerRow{{background:{bg};border:1px solid {border};"
            f"border-radius:3px;}}"
            f"QLabel{{color:{fg};font-size:12px;}}")

    def mousePressEvent(self, event):
        self.clicked.emit(self.key)


# ===================== 导出页面 =====================

class FrameExportPage(QWidget):

    PREVIEW_MAX = 1500       # 预览精灵图最长边像素
    PREVIEW_CACHE = 360      # 单帧预览缓存最长边

    def __init__(self, shared_state: dict):
        super().__init__()
        self.shared_state = shared_state
        self.setStyleSheet("QWidget{background:#2b2b2b;color:#ecf0f1;}")

        # 数据
        self.frames_dir = ""
        self.matted_dir = self.refined_dir = self.colored_dir = ""
        self.export_dir = ""
        self.items: List[dict] = []
        self._batch_id = ""
        self._batch_mark: Optional[float] = None
        self.preview_cache: Dict[str, np.ndarray] = {}
        self.selected_key: Optional[str] = None
        self.layer_rows: List[LayerRow] = []
        self.loader: Optional[_PreviewLoader] = None
        self.worker: Optional[_ExportWorker] = None

        self._build_ui()

        # 预览/持久化防抖
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._render_preview)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._write_info)

    # ---------- UI ----------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ===== 左侧：预览 =====
        left_box = QGroupBox("导出预览（棋盘格=透明）")
        lv = QVBoxLayout(left_box)
        self.view = FitImageLabel(checker=True)
        lv.addWidget(self.view, 1)
        self.info_label = QLabel("尚未加载图片")
        self.info_label.setStyleSheet("color:#f1c40f;font-size:12px;")
        lv.addWidget(self.info_label)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;font-size:12px;")
        lv.addWidget(self.status)
        root.addWidget(left_box, 13)

        # ===== 右侧：两个功能 Tab =====
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_sprite_tab(), "① 精灵图 PNG")
        self.tabs.addTab(self._build_psd_tab(), "② PSD 分层图")
        self.tabs.currentChanged.connect(lambda _i: self._render_preview())
        root.addWidget(self.tabs, 8)

    def _build_sprite_tab(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(8)

        g1 = QGroupBox("排列与分辨率")
        f = QVBoxLayout(g1)
        f.setSpacing(6)
        row1 = QHBoxLayout()
        self.chk_auto_cols = QCheckBox("自动列数（近方形）")
        self.chk_auto_cols.setChecked(True)
        self.chk_auto_cols.toggled.connect(self._on_setting_changed)
        row1.addWidget(self.chk_auto_cols)
        row1.addStretch()
        self.spn_cols = QSpinBox()
        self.spn_cols.setRange(1, 999)
        self.spn_cols.setValue(4)
        self.spn_cols.setEnabled(False)
        self.spn_cols.valueChanged.connect(self._on_setting_changed)
        self.chk_auto_cols.toggled.connect(
            lambda on: self.spn_cols.setEnabled(not on))
        row1.addWidget(QLabel("列数"))
        row1.addWidget(self.spn_cols)
        f.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("间距："))
        self.spn_pad = QSpinBox()
        self.spn_pad.setRange(0, 500)
        self.spn_pad.setValue(0)
        self.spn_pad.setSuffix(" px")
        self.spn_pad.valueChanged.connect(self._on_setting_changed)
        row2.addWidget(self.spn_pad)
        row2.addStretch()
        f.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("分辨率缩放："))
        self.sld_scale = QSlider(Qt.Horizontal)
        self.sld_scale.setRange(10, 400)
        self.sld_scale.setValue(100)
        self.spn_scale = QSpinBox()
        self.spn_scale.setRange(10, 400)
        self.spn_scale.setSuffix(" %")
        self.spn_scale.setValue(100)
        self.spn_scale.setFixedWidth(86)
        self.sld_scale.valueChanged.connect(self.spn_scale.setValue)
        self.spn_scale.valueChanged.connect(self.sld_scale.setValue)
        self.sld_scale.valueChanged.connect(self._on_setting_changed)
        row3.addWidget(self.sld_scale, 1)
        row3.addWidget(self.spn_scale)
        f.addLayout(row3)

        self.size_label = QLabel("输出尺寸：-")
        self.size_label.setStyleSheet("color:#2ecc71;")
        f.addWidget(self.size_label)
        v.addWidget(g1, 0)

        g2 = QGroupBox("导出")
        bv = QVBoxLayout(g2)
        self.btn_export_sprite = QPushButton("💾 导出精灵图 PNG")
        self.btn_export_sprite.clicked.connect(self._export_sprite)
        bv.addWidget(self.btn_export_sprite)
        btn_open1 = QPushButton("📂 打开导出目录")
        btn_open1.clicked.connect(self._open_export_dir)
        bv.addWidget(btn_open1)
        tip = QLabel("所有帧按序号排列拼为一张透明底图片，\n"
                     "可用于游戏引擎 / 动画合成。")
        tip.setStyleSheet("color:#888;font-size:11px;")
        tip.setWordWrap(True)
        bv.addWidget(tip)
        v.addWidget(g2, 0)
        v.addStretch(1)
        return page

    def _build_psd_tab(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(8)

        g1 = QGroupBox("画布与图层")
        f = QVBoxLayout(g1)
        f.setSpacing(6)
        self.canvas_label = QLabel("画布尺寸：-")
        self.canvas_label.setStyleSheet("color:#2ecc71;")
        f.addWidget(self.canvas_label)

        row = QHBoxLayout()
        self.chk_merged = QCheckBox("预览合并效果")
        self.chk_merged.toggled.connect(self._schedule_preview)
        row.addWidget(self.chk_merged)
        row.addStretch()
        row.addWidget(QLabel("统一缩放"))
        self.spn_all = QSpinBox()
        self.spn_all.setRange(5, 400)
        self.spn_all.setValue(100)
        self.spn_all.setSuffix(" %")
        self.spn_all.setFixedWidth(82)
        row.addWidget(self.spn_all)
        btn_apply_all = QPushButton("应用到全部")
        btn_apply_all.clicked.connect(self._apply_scale_all)
        row.addWidget(btn_apply_all)
        btn_reset_all = QPushButton("全部100%")
        btn_reset_all.clicked.connect(self._reset_scale_all)
        row.addWidget(btn_reset_all)
        f.addLayout(row)
        v.addWidget(g1, 0)

        g2 = QGroupBox("每个图层可单独缩放（点击行选中）")
        gl = QVBoxLayout(g2)
        gl.setContentsMargins(4, 6, 4, 4)
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setStyleSheet(
            "QScrollArea{background:#2b2b2b;border:none;}")
        self.layer_host = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_host)
        self.layer_layout.setContentsMargins(2, 2, 2, 2)
        self.layer_layout.setSpacing(3)
        self.layer_layout.addStretch()
        self.layer_scroll.setWidget(self.layer_host)
        gl.addWidget(self.layer_scroll)
        v.addWidget(g2, 1)

        g3 = QGroupBox("导出")
        bv = QVBoxLayout(g3)
        self.btn_export_psd = QPushButton("💾 导出 PSD 分层图")
        self.btn_export_psd.clicked.connect(self._export_psd)
        bv.addWidget(self.btn_export_psd)
        btn_open2 = QPushButton("📂 打开导出目录")
        btn_open2.clicked.connect(self._open_export_dir)
        bv.addWidget(btn_open2)
        v.addWidget(g3, 0)
        return page

    # ---------- 批次/加载 ----------

    def showEvent(self, event):
        super().showEvent(event)
        self._reload_if_changed()

    def _reload_if_changed(self):
        matted_dir = self.shared_state.get("matted_dir", "")
        frames_dir = self.shared_state.get("frames_dir", "")
        refined_dir = self.shared_state.get("refined_dir", "")
        colored_dir = self.shared_state.get("colored_dir", "")
        if refined_dir and not os.path.isdir(refined_dir):
            refined_dir = ""
        if colored_dir and not os.path.isdir(colored_dir):
            colored_dir = ""
        if not matted_dir or not os.path.isdir(matted_dir):
            self._show_empty("尚未找到制作好的图片\n\n请先完成批量抠图 / 精抠 / 色调统一")
            return
        files = sorted(f for f in os.listdir(matted_dir)
                       if f.lower().endswith(".png"))
        if not files:
            self._show_empty("批量抠图目录中没有 PNG 图片")
            return

        batch_id = "|".join((
            os.path.abspath(frames_dir) if frames_dir else "",
            os.path.abspath(matted_dir),
            self._sig(matted_dir), self._sig(refined_dir),
            self._sig(colored_dir)))
        if batch_id == self._batch_id:
            return
        batch_changed = bool(self._batch_id)
        self._batch_id = batch_id
        self._batch_mark = self._batch_mtime(matted_dir, refined_dir,
                                             colored_dir)
        new_export = os.path.join(frames_dir, "export") if frames_dir else \
            os.path.join(os.path.dirname(matted_dir), "export")
        preserved = None
        if batch_changed:
            preserved = self._discard_stale(self.export_dir, new_export)
        self.frames_dir, self.matted_dir = frames_dir, matted_dir
        self.refined_dir, self.colored_dir = refined_dir, colored_dir
        self.export_dir = new_export
        os.makedirs(self.export_dir, exist_ok=True)
        if preserved is not None:
            try:
                with open(self._info_path(), "w", encoding="utf-8") as f:
                    json.dump(preserved, f, ensure_ascii=False, indent=2)
            except OSError:
                pass

        self.items = []
        for i, fn in enumerate(files, 1):
            src, stage = self._best_source(fn, colored_dir,
                                           refined_dir, matted_dir)
            size = eu.image_size(src) or (0, 0)
            self.items.append({"key": fn, "number": i, "src": src,
                               "stage": stage, "w": size[0], "h": size[1]})

        self._restore_settings()
        if self.selected_key not in {it["key"] for it in self.items}:
            self.selected_key = self.items[0]["key"]
        self._rebuild_layer_list()
        self._start_preview_loader()
        self._update_size_info()
        self.status_text(
            f"已加载 {len(self.items)} 张（来源：{self._stage_name()}）")

    @staticmethod
    def _sig(directory: str) -> str:
        if not directory:
            return ""
        try:
            return "files:" + "|".join(sorted(
                f for f in os.listdir(directory)
                if f.lower().endswith(".png")))
        except OSError:
            return ""

    @staticmethod
    def _batch_mtime(*dirs) -> Optional[float]:
        marks = []
        for d in dirs:
            if d and os.path.isdir(d):
                for info in ("colored_info.json", "refined_info.json",
                             "matted_info.json"):
                    p = os.path.join(d, info)
                    if os.path.isfile(p):
                        try:
                            marks.append(os.path.getmtime(p))
                        except OSError:
                            pass
        return max(marks) if marks else None

    def _is_current(self, path: str) -> bool:
        if not path or not os.path.isfile(path):
            return False
        if self._batch_mark is None:
            return True
        try:
            return os.path.getmtime(path) >= self._batch_mark - 1e-6
        except OSError:
            return False

    def _best_source(self, fn: str, colored_dir: str, refined_dir: str,
                     matted_dir: str) -> Tuple[str, str]:
        """来源优先级 colored → refined → matted（需属当前批次）"""
        if colored_dir:
            p = os.path.join(colored_dir, fn)
            if self._is_current(p):
                return p, "colored"
        if refined_dir:
            p = os.path.join(refined_dir, fn)
            if os.path.isfile(p):
                return p, "refined"
        return os.path.join(matted_dir, fn), "matted"

    def _stage_name(self) -> str:
        stages = {it["stage"] for it in self.items}
        names = {"colored": "色调统一", "refined": "图片精抠",
                 "matted": "批量抠图"}
        return "、".join(names[s] for s in
                        ("colored", "refined", "matted") if s in stages)

    def _discard_stale(self, old_dir: str, new_dir: str) -> Optional[dict]:
        """批次变化时清理旧派生产物；export_info.json 是用户偏好，保留返回"""
        preserved = None
        if not old_dir or not new_dir:
            return preserved
        if os.path.basename(os.path.normpath(old_dir)).lower() != "export":
            return preserved
        if os.path.abspath(old_dir) == os.path.abspath(new_dir):
            info_p = os.path.join(old_dir, "export_info.json")
            if os.path.isfile(info_p):
                try:
                    with open(info_p, "r", encoding="utf-8") as f:
                        preserved = json.load(f)
                except OSError:
                    preserved = None
            shutil.rmtree(old_dir, ignore_errors=True)
        if os.path.abspath(self.shared_state.get("export_dir", "")) \
                == os.path.abspath(old_dir):
            self.shared_state["export_dir"] = ""
        return preserved

    def _show_empty(self, text: str):
        self.items, self.layer_rows = [], []
        self._batch_id = ""
        self.preview_cache = {}
        self.view.set_image(None)
        self.info_label.setText(text)
        self.size_label.setText("输出尺寸：-")
        self.canvas_label.setText("画布尺寸：-")
        self.btn_export_sprite.setEnabled(False)
        self.btn_export_psd.setEnabled(False)
        # 清空图层列表
        while self.layer_layout.count() > 1:
            it = self.layer_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    # ---------- 预览缓存加载 ----------

    def _start_preview_loader(self):
        if self.loader is not None and self.loader.isRunning():
            self.loader.stop()
            self.loader.wait(2000)
        self.preview_cache = {}
        self.loader = _PreviewLoader(self.items, self.PREVIEW_CACHE)
        self.loader.loaded_one.connect(self._on_preview_one)
        self.loader.done.connect(self._render_preview)
        self.loader.done.connect(
            lambda: self.status_text(
                f"预览就绪：{len(self.items)} 张（来源：{self._stage_name()}）"))
        self.loader.start()
        # 导出读的是全分辨率原文件，不依赖预览缓存，立即可用
        self.btn_export_sprite.setEnabled(True)
        self.btn_export_psd.setEnabled(True)

    def _on_preview_one(self, key: str, img: np.ndarray):
        self.preview_cache[key] = img
        if len(self.preview_cache) == 1 or len(self.preview_cache) % 5 == 0:
            self._render_preview()
        self.status_text(f"正在生成预览 {len(self.preview_cache)}"
                         f"/{len(self.items)}")

    # ---------- 设置 ----------

    def _sprite_params(self) -> dict:
        return {
            "auto_columns": self.chk_auto_cols.isChecked(),
            "columns": self.spn_cols.value(),
            "padding": self.spn_pad.value(),
            "sprite_scale": self.spn_scale.value(),
        }

    def _on_setting_changed(self):
        self._update_size_info()
        self._schedule_preview()
        self._save_timer.start()

    def _schedule_preview(self, *a):
        if self.preview_cache:
            self._preview_timer.start()

    def _update_size_info(self):
        if not self.items:
            self.size_label.setText("输出尺寸：-")
            self.canvas_label.setText("画布尺寸：-")
            return
        p = self._sprite_params()
        W, H, cols, rows = eu.sprite_sheet_size(
            [(it["w"], it["h"]) for it in self.items],
            columns=0 if p["auto_columns"] else p["columns"],
            scale=p["sprite_scale"] / 100.0,
            padding=p["padding"])
        self.size_label.setText(
            f"输出尺寸：{W} × {H} px（{cols} 列 × {rows} 行，"
            f"{len(self.items)} 帧）")
        cw = max(it["w"] for it in self.items)
        ch = max(it["h"] for it in self.items)
        self.canvas_label.setText(f"画布尺寸：{cw} × {ch} px（每帧居中）")

    # ---------- PSD 图层列表 ----------

    def _rebuild_layer_list(self):
        while self.layer_layout.count() > 1:
            it = self.layer_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.layer_rows = []
        for it in self.items:
            pct = int(self._layer_scales.get(it["key"], 100))
            row = LayerRow(it["key"], it["number"], it["w"], it["h"], pct)
            row.clicked.connect(self._select_layer)
            row.scale_changed.connect(self._on_layer_scale_changed)
            self.layer_layout.insertWidget(self.layer_layout.count() - 1, row)
            self.layer_rows.append(row)
        self._sync_row_selection()

    def _select_layer(self, key: str):
        self.selected_key = key
        self.chk_merged.setChecked(False)
        self._sync_row_selection()
        self._render_preview()

    def _sync_row_selection(self):
        for r in self.layer_rows:
            r.set_selected(r.key == self.selected_key)

    def _on_layer_scale_changed(self, key: str, pct: int):
        self._layer_scales[key] = pct
        self._schedule_preview()
        self._save_timer.start()

    def _apply_scale_all(self):
        v = self.spn_all.value()
        for r in self.layer_rows:
            r.spin.setValue(v)      # 触发 changed → 写 dict / 刷新
        self.status_text(f"已把全部图层缩放设为 {v}%")

    def _reset_scale_all(self):
        for r in self.layer_rows:
            r.spin.setValue(100)
        self.spn_all.setValue(100)
        self.status_text("全部图层缩放已恢复 100%")

    # ---------- 预览渲染 ----------

    def _render_preview(self):
        if not self.items:
            self.view.set_image(None)
            return
        if self.tabs.currentIndex() == 0:
            self._render_sprite_preview()
        else:
            self._render_psd_preview()

    def _ready_images(self) -> List[Tuple[dict, np.ndarray]]:
        return [(it, self.preview_cache[it["key"]])
                for it in self.items if it["key"] in self.preview_cache]

    def _render_sprite_preview(self):
        ready = self._ready_images()
        if not ready:
            return
        p = self._sprite_params()
        user_scale = p["sprite_scale"] / 100.0
        # 用缓存小图先按 1:1 结算基础尺寸，再整体等比收进预览上限
        base_sizes = [(im.shape[1], im.shape[0]) for _, im in ready]
        W0, H0, _, _ = eu.sprite_sheet_size(
            base_sizes,
            columns=0 if p["auto_columns"] else p["columns"],
            scale=1.0, padding=p["padding"])
        pf = min(1.0, self.PREVIEW_MAX
                 / max(1.0, max(W0, H0) * user_scale))
        sheet, _ = eu.build_sprite_sheet(
            [im for _, im in ready],
            columns=0 if p["auto_columns"] else p["columns"],
            scale=user_scale * pf, padding=max(1, int(p["padding"] * pf)))
        self.view.set_image(bgra_to_pixmap(sheet))
        self.info_label.setText(
            f"精灵图预览（{len(ready)}/{len(self.items)} 帧，"
            f"预览缩放 {user_scale * pf:.2f}×）")

    def _render_psd_preview(self):
        ready = self._ready_images()
        if not ready:
            return
        cw = max(im.shape[1] for _, im in ready)
        ch = max(im.shape[0] for _, im in ready)
        canvas = np.zeros((ch, cw, 4), np.uint8)
        if self.chk_merged.isChecked():
            for it, im in ready:
                pct = self._layer_scales.get(it["key"], 100) / 100.0
                sim = eu.scale_bgra(im, pct)
                x, y = eu.centered_xy(cw, ch, sim.shape[1], sim.shape[0])
                eu.flatten_over(canvas, sim, x, y)
            what = "全部图层合并效果"
        else:
            it = next((x for x in self.items
                       if x["key"] == self.selected_key), None)
            if it is None or it["key"] not in self.preview_cache:
                it, im = ready[0]
            else:
                im = self.preview_cache[it["key"]]
            pct = self._layer_scales.get(it["key"], 100) / 100.0
            sim = eu.scale_bgra(im, pct)
            x, y = eu.centered_xy(cw, ch, sim.shape[1], sim.shape[0])
            eu.flatten_over(canvas, sim, x, y)
            what = f"当前图层：{it['key']}（{int(round(pct*100))}%）"
        self.view.set_image(bgra_to_pixmap(canvas))
        self.info_label.setText(
            f"PSD 预览：{what}（{len(ready)}/{len(self.items)} 帧已缓存）")

    # ---------- 导出 ----------

    def _export(self, mode: str):
        if not self.items or self.worker is not None:
            return
        name = "sprite_sheet.png" if mode == "sprite" else "layers.psd"
        out_path = os.path.join(self.export_dir, name)
        params = self._sprite_params()
        params["layer_scales"] = dict(self._layer_scales)
        self.worker = _ExportWorker(mode, self.items, params, out_path)
        self.worker.progress.connect(self.status_text)
        self.worker.failed.connect(self._on_export_failed)
        self.worker.finished_ok.connect(self._on_export_ok)
        self.worker.finished.connect(self._on_worker_end)
        self.btn_export_sprite.setEnabled(False)
        self.btn_export_psd.setEnabled(False)
        self.worker.start()

    def _export_sprite(self):
        self._export("sprite")

    def _export_psd(self):
        self._export("psd")

    def _on_export_ok(self, mode: str, path: str):
        self.shared_state["export_dir"] = self.export_dir
        self._write_info()
        self.status_text(f"✅ 已导出：{path}")

    def _on_export_failed(self, msg: str):
        self.status_text(f"导出失败：{msg}")
        QMessageBox.warning(self, "导出失败", msg)

    def _on_worker_end(self):
        self.worker = None
        self.btn_export_sprite.setEnabled(True)
        self.btn_export_psd.setEnabled(True)

    def _open_export_dir(self):
        if self.export_dir and os.path.isdir(self.export_dir):
            os.startfile(os.path.abspath(self.export_dir))
        else:
            QMessageBox.information(self, "打开导出目录",
                                    "请先加载图片后再导出。")

    def status_text(self, text: str):
        self.status.setText(text)

    # ---------- 持久化 ----------

    def _info_path(self) -> str:
        return os.path.join(self.export_dir, "export_info.json")

    def _restore_settings(self):
        data = {}
        try:
            with open(self._info_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        self.chk_auto_cols.setChecked(bool(data.get("auto_columns", True)))
        self.spn_cols.setValue(int(data.get("columns", 4)))
        self.spn_pad.setValue(int(data.get("padding", 0)))
        self.spn_scale.setValue(max(10, min(400,
                                            int(data.get("sprite_scale", 100)))))
        saved_scales = data.get("layer_scales", {})
        keys = {it["key"] for it in self.items}
        self._layer_scales = {
            k: max(5, min(400, int(v)))
            for k, v in saved_scales.items() if k in keys}

    def _write_info(self):
        if not self.export_dir or not self.items:
            return
        os.makedirs(self.export_dir, exist_ok=True)
        data = {
            "auto_columns": self.chk_auto_cols.isChecked(),
            "columns": self.spn_cols.value(),
            "padding": self.spn_pad.value(),
            "sprite_scale": self.spn_scale.value(),
            "merged_preview": self.chk_merged.isChecked(),
            "layer_scales": self._layer_scales,
        }
        try:
            with open(self._info_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
