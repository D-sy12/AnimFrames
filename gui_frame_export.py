# -*- coding: utf-8 -*-
"""
模块六：导出图片 - 页面（可嵌入主窗口）

布局（左右固定 13:8）：
  左侧：PS 式工作台画布（棋盘格透明底），实时合成显示所有"睁眼"图层，
        也可切换为精灵图拼接预览。
  右侧上半：画布与工具（移动/缩放/框选、视图缩放、画布/图像大小、
        导入图片）+ 精灵图 PNG 参数（行数、间距、导出）。
  右侧下半：所有图片按一列排列，每行 = 小眼睛(是否在工作台显示/参与
        导出) + 缩略图 + 名称尺寸 + 该图层独立缩放%。

功能：
  动画精灵图：可见帧拼一张透明底 PNG，自动/按目标行数排列、间距、
  每张图片独立缩放（真实重采样进入导出）。支持 PS 式画布大小 /
  图像大小对话框、框选多图层整组移动缩放、每图独立撤销（Ctrl+Z）。

图片来源优先级：色调统一 colored/ → 精抠 refined/ → 批量抠图 matted/。
产物目录：frames_dir/export/（sprite_sheet.png）。
"""

import os
import json
import time
import shutil
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal, QThread, QRectF
from PySide6.QtGui import QKeySequence, QShortcut, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QScrollArea, QGroupBox, QCheckBox, QSpinBox,
    QToolButton, QFrame, QMessageBox, QFileDialog, QDialog,
    QDialogButtonBox, QComboBox, QDoubleSpinBox, QButtonGroup,
    QRadioButton, QSlider,
)

import export_utils as eu
from gui_frame_matting import (
    imread_unicode, imwrite_unicode, compose_checkerboard,
)
from gui_frame_color import FitImageLabel, bgra_to_pixmap


# ===================== 可交互工作台画布 =====================

class WorkspaceCanvas(FitImageLabel):
    """PS 式工作台：记录显示区域，把鼠标拖拽换算成全分辨率画布坐标增量"""

    drag_started = Signal()             # 左键按下（拖拽事务开始）
    drag_delta = Signal(float, float)   # (dx, dy) 全分辨率画布像素
    drag_finished = Signal()
    resized = Signal()
    marquee_changed = Signal(QRectF, bool)  # 画布坐标矩形, 进行中
    marquee_done = Signal(QRectF, bool)     # 框选结束, additive=Shift

    def __init__(self, checker: bool = True):
        super().__init__(checker)
        self._disp_rect = None           # 图像在控件中的显示矩形
        self.canvas_wh = (1, 1)          # 当前画布全分辨率宽高（换算用）
        self._last = None
        self.tool = "move"               # "move" / "scale"
        # 工作台分层：below / selected / above 三张 pixmap，
        # 拖拽时只换 selected，合成交给 QPainter（C++ 光栅化快路径）
        self._layer_below = None
        self._layer_above = None
        self._layer_sel = None
        self._sel_rect = None            # (x,y,w,h) 预览画布坐标
        self._rect_key = None
        self._scaled_size = None
        self._checker_tile = None
        # 视图缩放：滚轮以鼠标为锚缩放，中键拖拽平移；1.0 = 适配窗口
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_last = None
        self.zoom_changed = None         # 由页面赋值为回调(percent:int)
        # 框选：marquee 工具下的屏幕坐标起止点
        self._marquee_start = None
        self._marquee_cur = None
        self.setCursor(Qt.OpenHandCursor)

    def set_tool(self, tool: str):
        self.tool = tool
        self.setCursor(
            Qt.OpenHandCursor if tool == "move"
            else Qt.SizeAllCursor if tool == "scale"
            else Qt.CrossCursor)         # marquee

    def set_image(self, pix):
        """普通（精灵图预览）模式：退出分层模式，并复位视图缩放"""
        self._layer_below = None
        self._layer_above = None
        self._layer_sel = None
        self._sel_rect = None
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._notify_zoom()
        super().set_image(pix)

    def set_workspace_layers(self, anchor, below, above,
                             sel_pm, sel_rect):
        self._layer_below = below
        self._layer_above = above
        self._layer_sel = sel_pm
        self._sel_rect = sel_rect
        self._pix = anchor               # 仅用于几何/坐标换算
        self._rect_key = None
        self.update()

    def resizeEvent(self, e):
        if self._layer_below is not None:
            self._rect_key = None
            self.update()
        else:
            super().resizeEvent(e)
        self.resized.emit()

    def _target_rect(self):
        """预览画布在控件中的等比显示矩形（QRectF），并刷新 _disp_rect"""
        if self._pix is None or self._pix.isNull():
            self._disp_rect = None
            return None
        pw, ph = self._pix.width(), self._pix.height()
        key = (self.width(), self.height(), pw, ph)
        if key != self._rect_key:
            scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                      Qt.FastTransformation)
            self._rect_key = key
            self._scaled_size = (scaled.width(), scaled.height())
        sw, sh = self._scaled_size
        sw, sh = sw * self._zoom, sh * self._zoom
        x = (self.width() - sw) / 2 + self._pan_x
        y = (self.height() - sh) / 2 + self._pan_y
        self._disp_rect = (int(x), int(y), int(sw), int(sh))
        return QRectF(x, y, sw, sh)

    # ---------- 视图缩放/平移 ----------

    ZOOM_MIN, ZOOM_MAX = 0.05, 20.0

    def _notify_zoom(self):
        if self.zoom_changed:
            self.zoom_changed(int(round(self._zoom * 100)))

    def zoom_reset(self):
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._rect_key = None
        self.update()
        self._notify_zoom()

    def zoom_step(self, factor: float):
        """按钮缩放：以视图中心为锚"""
        self._apply_zoom(self._zoom * factor,
                         self.width() / 2, self.height() / 2)

    def _apply_zoom(self, new_zoom: float, mx: float, my: float):
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, new_zoom))
        if new_zoom == self._zoom:
            return
        rect = self._target_rect()
        if rect is None:
            return
        k = new_zoom / self._zoom
        cx, cy = rect.x(), rect.y()
        # 保持鼠标下的画布点不动：rect 左上角相对鼠标的距离按比例伸缩
        self._pan_x = mx - (mx - cx) * k - (self.width() - rect.width() * k) / 2
        self._pan_y = my - (my - cy) * k - (self.height() - rect.height() * k) / 2
        self._zoom = new_zoom
        self._rect_key = None
        self.update()
        self._notify_zoom()

    def wheelEvent(self, e):
        if self._pix is not None and not self._pix.isNull():
            factor = 1.25 if e.angleDelta().y() > 0 else 0.8
            pos = e.position()
            self._apply_zoom(self._zoom * factor, pos.x(), pos.y())
            e.accept()
            return
        super().wheelEvent(e)

    def paintEvent(self, event):
        if self._layer_below is None and \
                (self._pix is None or self._pix.isNull()):
            super().paintEvent(event)
            return
        rect = self._target_rect()
        if rect is None:
            return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#222"))
        p.setClipRect(rect)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._checker:
            if self._checker_tile is None:
                from gui_frame_matting import _checker_tile
                self._checker_tile = _checker_tile(12)
            p.drawTiledPixmap(rect, self._checker_tile)
        if self._layer_below is None:
            # 普通模式（精灵图预览）：单张 _pix，同样支持缩放/平移
            p.drawPixmap(rect, self._pix,
                         QRectF(0, 0, self._pix.width(), self._pix.height()))
            p.end()
            return
        pw = self._layer_below.width()
        ph = self._layer_below.height()
        full = QRectF(0, 0, pw, ph)
        p.drawPixmap(rect, self._layer_below, full)
        if self._layer_sel is not None and self._sel_rect is not None:
            sx, sy, sw, sh = self._sel_rect
            kx, ky = rect.width() / pw, rect.height() / ph
            p.drawPixmap(
                QRectF(rect.x() + sx * kx, rect.y() + sy * ky,
                       sw * kx, sh * ky),
                self._layer_sel, QRectF(0, 0, sw, sh))
        if self._layer_above is not None:
            p.drawPixmap(rect, self._layer_above, full)
        # 框选橡皮筋（屏幕坐标，半透明填充+白虚线）
        if self._marquee_start is not None and self._marquee_cur is not None:
            from PySide6.QtGui import QPen
            from PySide6.QtCore import QPointF as _PF
            s, c = self._marquee_start, self._marquee_cur
            band = QRectF(_PF(min(s.x(), c.x()), min(s.y(), c.y())),
                          _PF(max(s.x(), c.x()), max(s.y(), c.y())))
            p.setClipping(False)
            p.fillRect(band, QColor(80, 160, 255, 50))
            pen = QPen(QColor("#ffffff"))
            pen.setStyle(Qt.DashLine)
            pen.setWidth(1)
            p.setPen(pen)
            p.drawRect(band)
        p.end()

    def _render(self):
        # 两种模式统一走 paintEvent 自绘（支持视图缩放/平移）
        if self._pix is None or self._pix.isNull():
            self._disp_rect = None
            self.clear()
            self.update()
            return
        self._rect_key = None
        self.update()

    def _screen_to_canvas_rect(self, p0, p1) -> QRectF:
        """屏幕矩形（两个任意角点）→ 预览画布坐标矩形"""
        x, y, dw, dh = self._disp_rect
        pw, ph = self._pix.width(), self._pix.height()
        x0 = (min(p0.x(), p1.x()) - x) * pw / dw
        x1 = (max(p0.x(), p1.x()) - x) * pw / dw
        y0 = (min(p0.y(), p1.y()) - y) * ph / dh
        y1 = (max(p0.y(), p1.y()) - y) * ph / dh
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton and self._disp_rect:
            self._pan_last = e.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.LeftButton and self._disp_rect:
            if self.tool == "marquee":
                self._marquee_start = e.position()
                self._marquee_cur = e.position()
                self.update()
            else:
                self._last = e.position()
                self.setCursor(Qt.ClosedHandCursor
                               if self.tool == "move" else Qt.SizeAllCursor)
                self.drag_started.emit()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan_last is not None:
            pos = e.position()
            self._pan_x += pos.x() - self._pan_last.x()
            self._pan_y += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self._rect_key = None
            self.update()
        elif self._marquee_start is not None:
            self._marquee_cur = e.position()
            self.update()
            self.marquee_changed.emit(
                self._screen_to_canvas_rect(self._marquee_start,
                                            self._marquee_cur),
                bool(e.modifiers() & Qt.ShiftModifier))
        elif self._last is not None and self._disp_rect:
            pos = e.position()
            dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
            self._last = pos
            dw = self._disp_rect[2]
            if dw > 0:
                f = self.canvas_wh[0] / dw     # 等比缩放，fx=fy
                self.drag_delta.emit(dx * f, dy * f)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self.set_tool(self.tool)
        elif self._marquee_start is not None and e.button() == Qt.LeftButton:
            rect = self._screen_to_canvas_rect(self._marquee_start,
                                               e.position())
            self._marquee_start = None
            self._marquee_cur = None
            self.update()
            self.marquee_done.emit(
                rect, bool(e.modifiers() & Qt.ShiftModifier))
        elif self._last is not None:
            self._last = None
            self.set_tool(self.tool)
            self.drag_finished.emit()
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and \
                (self._zoom != 1.0 or self._pan_x or self._pan_y):
            self.zoom_reset()
        super().mouseDoubleClickEvent(e)


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
            self._run_sprite()
        except Exception as e:
            self.failed.emit(str(e))

    def _load(self, item: dict) -> np.ndarray:
        img = eu.load_bgra(item["src"])
        if img is None:
            raise IOError(f"读取失败：{item['src']}")
        return img

    def _visible(self, it: dict) -> bool:
        return self.params["layer_visible"].get(it["key"], True)

    def _run_sprite(self):
        p = self.params
        items = [it for it in self.items if self._visible(it)]
        if not items:
            raise ValueError("没有可见图片（全部被小眼睛隐藏）")
        images = []
        for i, it in enumerate(items, 1):
            img = self._load(it)
            # 单张图片缩放（layer_scales）生效：按各自图层缩放重采样
            layer_scale = p["layer_scales"].get(it["key"], 100) / 100.0
            img = eu.scale_bgra(img, layer_scale)
            images.append(img)
            self.progress.emit(f"读取图片 {i}/{len(items)}")
        self.progress.emit("正在拼接精灵图…")
        canvas, _ = eu.build_sprite_sheet(
            images, columns=p["columns"],
            scale=1.0, padding=p["padding"])
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        if not imwrite_unicode(self.out_path, canvas):
            raise IOError(f"写入失败：{self.out_path}")
        self.progress.emit(
            f"精灵图已导出：{os.path.basename(self.out_path)} "
            f"{canvas.shape[1]}×{canvas.shape[0]}（{len(items)} 帧）")
        self.finished_ok.emit("sprite", self.out_path)


class _PreviewLoader(QThread):
    """把所有帧解码并缩小缓存，避免预览每次读全分辨率"""
    loaded_one = Signal(str, object, float)   # key, 缓存BGRA, 缓存/原尺寸系数
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
            self.loaded_one.emit(it["key"], img, f)
        self.done.emit()


# ===================== PS 风格「画布大小 / 图像大小」对话框 =====================

class CanvasSizeDialog(QDialog):
    """仿 PS「画布大小」：当前/新建宽高 + 相对 + 九宫格定位 + 扩展颜色"""

    def __init__(self, parent, cur_w: int, cur_h: int,
                 new_w: int, new_h: int, anchor: Tuple[int, int] = (1, 1),
                 ext_color: str = "透明"):
        super().__init__(parent)
        self.setWindowTitle("画布大小")
        self.setModal(True)
        self.setMinimumWidth(360)
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        # 当前大小（只读）
        cur_box = QGroupBox("当前大小")
        cf = QGridLayout(cur_box)
        cf.addWidget(QLabel("宽度："), 0, 0)
        cf.addWidget(QLabel(f"{cur_w} 像素"), 0, 1)
        cf.addWidget(QLabel("高度："), 1, 0)
        cf.addWidget(QLabel(f"{cur_h} 像素"), 1, 1)
        lay.addWidget(cur_box)

        # 新建大小
        new_box = QGroupBox("新建大小")
        nf = QGridLayout(new_box)
        nf.addWidget(QLabel("宽度(W)："), 0, 0)
        self.spn_w = QSpinBox()
        self.spn_w.setRange(1, 16384)
        self.spn_w.setValue(new_w)
        self.spn_w.setSuffix(" 像素")
        nf.addWidget(self.spn_w, 0, 1)
        nf.addWidget(QLabel("高度(H)："), 1, 0)
        self.spn_h = QSpinBox()
        self.spn_h.setRange(1, 16384)
        self.spn_h.setValue(new_h)
        self.spn_h.setSuffix(" 像素")
        nf.addWidget(self.spn_h, 1, 1)
        self.chk_relative = QCheckBox("相对(R)")
        self.chk_relative.setToolTip(
            "勾选后按相对增量调整（如 +100 / -50 像素）")
        nf.addWidget(self.chk_relative, 2, 0, 1, 2)
        self._base_w, self._base_h = new_w, new_h
        self.chk_relative.toggled.connect(self._on_relative)
        lay.addWidget(new_box)

        # 定位（九宫格锚点）
        anchor_box = QGroupBox("定位")
        af = QGridLayout(anchor_box)
        self.anchor_btns: List[List[QToolButton]] = []
        self._anchor = list(anchor)          # (row, col) 0..2
        arrows = [["↖", "↑", "↗"], ["←", "·", "→"], ["↙", "↓", "↘"]]
        for r in range(3):
            row = []
            for c in range(3):
                b = QToolButton()
                b.setText(arrows[r][c])
                b.setCheckable(True)
                b.setFixedSize(26, 26)
                b.clicked.connect(
                    lambda _=False, rr=r, cc=c: self._set_anchor(rr, cc))
                af.addWidget(b, r, c)
                row.append(b)
            self.anchor_btns.append(row)
        self._refresh_anchor()
        lay.addWidget(anchor_box)

        # 画布扩展颜色
        rowc = QHBoxLayout()
        rowc.addWidget(QLabel("画布扩展颜色："))
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["透明", "白色", "黑色"])
        self.cmb_color.setCurrentText(ext_color)
        rowc.addWidget(self.cmb_color, 1)
        lay.addLayout(rowc)

        btns = QDialogButtonBox(QDialogButtonBox.Ok
                                | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _on_relative(self, on: bool):
        if on:
            self._base_w, self._base_h = self.spn_w.value(), \
                self.spn_h.value()
            # 相对模式允许 0 和负增量
            self.spn_w.setRange(-16384, 16384)
            self.spn_h.setRange(-16384, 16384)
            self.spn_w.setValue(0)
            self.spn_h.setValue(0)
        else:
            # 先按增量算出目标绝对值（此时仍是宽范围，可承载负增量）
            tw = max(1, self._base_w + self.spn_w.value())
            th = max(1, self._base_h + self.spn_h.value())
            self.spn_w.setRange(1, 16384)
            self.spn_h.setRange(1, 16384)
            self.spn_w.setValue(tw)
            self.spn_h.setValue(th)

    def _set_anchor(self, r: int, c: int):
        self._anchor = [r, c]
        self._refresh_anchor()

    def _refresh_anchor(self):
        for r in range(3):
            for c in range(3):
                self.anchor_btns[r][c].setChecked(
                    [r, c] == self._anchor)

    def result_values(self):
        w, h = self.spn_w.value(), self.spn_h.value()
        if self.chk_relative.isChecked():
            w = self._base_w + w
            h = self._base_h + h
        return (max(1, min(16384, w)), max(1, min(16384, h)),
                tuple(self._anchor), self.cmb_color.currentText())


class ImageSizeDialog(QDialog):
    """仿 PS「图像大小」：像素尺寸/百分比，联动整幅画布+所有图层"""

    def __init__(self, parent, cur_w: int, cur_h: int, pct: int = 100):
        super().__init__(parent)
        self.setWindowTitle("图像大小")
        self.setModal(True)
        self.setMinimumWidth(340)
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        size_box = QGroupBox("像素尺寸")
        sf = QGridLayout(size_box)
        sf.addWidget(QLabel(f"当前：{cur_w} × {cur_h} 像素"), 0, 0, 1, 4)
        sf.addWidget(QLabel("宽度(W)："), 1, 0)
        self.spn_w = QSpinBox()
        self.spn_w.setRange(1, 16384)
        self.spn_w.setValue(cur_w)
        self.spn_w.setSuffix(" 像素")
        sf.addWidget(self.spn_w, 1, 1)
        sf.addWidget(QLabel("高度(H)："), 2, 0)
        self.spn_h = QSpinBox()
        self.spn_h.setRange(1, 16384)
        self.spn_h.setValue(cur_h)
        self.spn_h.setSuffix(" 像素")
        sf.addWidget(self.spn_h, 2, 1)
        self.chk_constrain = QCheckBox("约束比例")
        self.chk_constrain.setChecked(True)
        sf.addWidget(self.chk_constrain, 3, 0, 1, 4)
        self._ratio = cur_h / cur_w if cur_w else 1.0
        self._base_w = cur_w                 # 100% 基准（用于百分比换算）
        self._base_h = cur_h
        self.spn_w.valueChanged.connect(self._on_w)
        self.spn_h.valueChanged.connect(self._on_h)
        self._lock = False
        lay.addWidget(size_box)

        # 百分比
        rowp = QHBoxLayout()
        rowp.addWidget(QLabel("缩放百分比："))
        self.spn_pct = QDoubleSpinBox()
        self.spn_pct.setRange(1, 2000)
        self.spn_pct.setDecimals(1)
        self.spn_pct.setValue(100.0)
        self.spn_pct.setSuffix(" %")
        self.spn_pct.valueChanged.connect(self._on_pct)
        rowp.addWidget(self.spn_pct, 1)
        lay.addLayout(rowp)

        btns = QDialogButtonBox(QDialogButtonBox.Ok
                                | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _on_w(self, v: int):
        if self._lock or not self.chk_constrain.isChecked():
            return
        self._lock = True
        self.spn_h.setValue(max(1, int(round(v * self._ratio))))
        self._lock = False
        self._sync_pct()

    def _on_h(self, v: int):
        if self._lock or not self.chk_constrain.isChecked():
            return
        self._lock = True
        self.spn_w.setValue(max(1, int(round(v / self._ratio))))
        self._lock = False
        self._sync_pct()

    def _on_pct(self, p: float):
        # 百分比驱动像素尺寸（保持比例）
        if self._lock:
            return
        self._lock = True
        self.spn_w.setValue(max(1, int(round(self._base_w * p / 100))))
        self.spn_h.setValue(max(1, int(round(self._base_h * p / 100))))
        self._lock = False

    def _sync_pct(self):
        if self._lock:
            return
        self._lock = True
        self.spn_pct.setValue(self.spn_w.value() / self._base_w * 100)
        self._lock = False

    def result_scale(self) -> float:
        return self.spn_w.value() / self._base_w if self._base_w else 1.0


# ===================== 图层行（眼睛+缩略图+名称+缩放） =====================

class LayerRow(QFrame):
    """所有图片列表中的一行：👁 显示开关 + 缩略图 + 名称尺寸 + 撤销 + 独立缩放%"""
    clicked = Signal(str)
    scale_changed = Signal(str, int)
    visibility_changed = Signal(str, bool)
    undo_clicked = Signal(str)

    def __init__(self, key: str, number: int, w: int, h: int,
                 pct: int, visible: bool = True):
        super().__init__()
        self.key = key
        self._selected = False
        self._marqueed = False           # 框选多选标记（☑）
        self._visible = True
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(6)

        # 小眼睛：是否在工作台显示 / 参与导出
        self.eye = QToolButton()
        self.eye.setCheckable(True)
        self.eye.setFixedSize(28, 28)
        self.eye.setCursor(Qt.PointingHandCursor)
        self.eye.setToolTip("显示 / 隐藏该图片（隐藏后不参与导出）")
        self.eye.setStyleSheet(
            "QToolButton{background:transparent;border:none;"
            "font-size:15px;color:#ecf0f1;}"
            "QToolButton:hover{background:#3a3a3a;border-radius:4px;}")
        self.eye.toggled.connect(self._on_eye)
        lay.addWidget(self.eye)

        # 多选复选框（框选/手动多选）
        self.chk = QCheckBox()
        self.chk.setCursor(Qt.PointingHandCursor)
        self.chk.setToolTip("选中该图层（多选）")
        self.chk.stateChanged.connect(
            lambda s: self._on_marquee_toggle(bool(s)))
        lay.addWidget(self.chk)

        # 缩略图
        self.thumb = QLabel()
        self.thumb.setFixedSize(48, 36)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet(
            "background:#1a1a1a;border:1px solid #444;border-radius:2px;")
        lay.addWidget(self.thumb)

        self.name_lab = QLabel(f"#{number:03d}  {key}\n{w}×{h}")
        self.name_lab.setStyleSheet("background:transparent;border:none;")
        self.name_lab.setToolTip(key)
        lay.addWidget(self.name_lab, 1)

        # 本张图片独立撤销（移动/缩放）
        self.btn_undo = QToolButton()
        self.btn_undo.setText("↩")
        self.btn_undo.setFixedSize(28, 28)
        self.btn_undo.setCursor(Qt.PointingHandCursor)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setStyleSheet(
            "QToolButton{background:transparent;border:none;"
            "font-size:15px;color:#ecf0f1;}"
            "QToolButton:hover{background:#3a3a3a;border-radius:4px;}"
            "QToolButton:disabled{color:#555;}")
        self.btn_undo.clicked.connect(
            lambda: self.undo_clicked.emit(self.key))
        lay.addWidget(self.btn_undo)

        self.spin = QSpinBox()
        self.spin.setRange(5, 400)
        self.spin.setValue(pct)
        self.spin.setSuffix(" %")
        self.spin.setFixedWidth(78)
        self.spin.setCursor(Qt.ArrowCursor)
        self.spin.setToolTip("该图层独立缩放")
        self.spin.valueChanged.connect(
            lambda v: self.scale_changed.emit(self.key, int(v)))
        lay.addWidget(self.spin)

        self.set_visible(visible, emit_signal=False)
        self._set_style()

    def _on_eye(self, checked: bool):
        self.set_visible(checked, emit_signal=True)

    def set_visible(self, visible: bool, emit_signal: bool = False):
        self._visible = visible
        self.eye.blockSignals(True)
        self.eye.setChecked(visible)
        self.eye.blockSignals(False)
        self.eye.setText("👁" if visible else "—")
        op = "1.0" if visible else "0.45"
        self.name_lab.setStyleSheet(
            f"background:transparent;border:none;opacity:{op};")
        if emit_signal:
            self.visibility_changed.emit(self.key, visible)

    def is_visible(self) -> bool:
        return self._visible

    def set_undo_count(self, n: int):
        self.btn_undo.setEnabled(n > 0)
        self.btn_undo.setToolTip(
            f"撤销该图片上一次移动/缩放（可撤销 {n} 步）" if n
            else "该图片暂无可撤销的移动/缩放")

    def set_thumbnail(self, bgra: np.ndarray):
        pm = bgra_to_pixmap(bgra)
        if pm is None or pm.isNull():
            return
        self.thumb.setPixmap(pm.scaled(
            self.thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_marquee_toggle(self, checked: bool):
        if checked != self._marqueed:
            self._marqueed = checked
            self._set_style()

    def set_marqueed(self, m: bool):
        if m != self._marqueed:
            self._marqueed = m
            self.chk.blockSignals(True)
            self.chk.setChecked(m)
            self.chk.blockSignals(False)
            self._set_style()

    def set_selected(self, sel: bool):
        self._selected = sel
        self._set_style()

    def _set_style(self):
        if self._selected:
            border, bg = "#f1c40f", "#1a3a5c"      # 主选：金框
        elif self._marqueed:
            border, bg = "#5dade2", "#16324f"      # 框选：蓝框
        else:
            border, bg = "#444444", "#1e1e1e"
        self.setStyleSheet(
            f"LayerRow{{background:{bg};border:1px solid {border};"
            f"border-radius:3px;}}")

    def mousePressEvent(self, event):
        self.clicked.emit(self.key)


# ===================== 导出页面 =====================

class FrameExportPage(QWidget):

    PREVIEW_MAX = 1500       # 预览最长边像素
    PREVIEW_CACHE = 1000     # 单帧预览缓存最长边（过小于此会放大模糊）

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
        self.preview_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        # 各图非透明像素包围盒（全分辨率 l,t,r,b，右/下开区间），供框选命中
        self._alpha_bbox: Dict[str, Tuple[int, int, int, int]] = {}
        self.selected_key: Optional[str] = None
        self.selected_keys: set = set()          # 框选/多选的图层集合
        self.layer_rows: List[LayerRow] = []
        self._layer_scales: Dict[str, int] = {}
        self._layer_visible: Dict[str, bool] = {}
        self._layer_offsets: Dict[str, List[float]] = {}   # 相对居中偏移
        # 每张图片独立撤销栈：key -> [(缩放%, ox, oy), ...]（变更前的状态）
        self._undo: Dict[str, List[Tuple[int, float, float]]] = {}
        self._undo_coalesce: Dict[str, float] = {}         # spin 连调合并时间戳
        self._drag_pending: Optional[str] = None           # 拖拽事务待入栈的 key
        self._ws_cache: Optional[dict] = None              # 静态图层合成缓存
        self.loader: Optional[_PreviewLoader] = None
        self.worker: Optional[_ExportWorker] = None

        # 预览/持久化防抖（须在 _build_ui 前创建，画布连接用到）
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(24)
        self._preview_timer.timeout.connect(self._render_preview)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._write_info)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._on_view_resized)

        # 帧播放（检查抖动）：_play_key=None 表示正常显示全部图层
        self._play_key: Optional[str] = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        self._build_ui()
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo_selected)

    # ---------- UI ----------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ===== 左侧：PS 式工作台 =====
        left_box = QGroupBox("工作台（棋盘格=透明，选中图片后用工具拖拽）")
        lv = QVBoxLayout(left_box)
        self.view = WorkspaceCanvas(checker=True)
        self.view.drag_started.connect(self._on_drag_started)
        self.view.drag_delta.connect(self._on_canvas_drag)
        self.view.drag_finished.connect(self._on_drag_finished)
        self.view.resized.connect(self._resize_timer.start)
        self.view.marquee_done.connect(self._on_marquee_done)
        self.view.marquee_changed.connect(self._on_marquee_changed)
        lv.addWidget(self.view, 1)

        # ===== 帧播放（检查播放时各帧位置是否抖动）=====
        play_box = QGroupBox("帧播放（逐帧播放，检查主体位置是否抖动）")
        pl = QVBoxLayout(play_box)
        pl.setContentsMargins(8, 4, 8, 6)
        pl.setSpacing(4)
        pr1 = QHBoxLayout()
        pr1.setSpacing(4)
        self.btn_play_first = QToolButton()
        self.btn_play_first.setText("⏮")
        self.btn_play_first.setToolTip("第一帧")
        self.btn_play_prev = QToolButton()
        self.btn_play_prev.setText("◀")
        self.btn_play_prev.setToolTip("上一帧（暂停）")
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setFixedWidth(88)
        self.btn_play_next = QToolButton()
        self.btn_play_next.setText("▶")
        self.btn_play_next.setToolTip("下一帧（暂停）")
        self.btn_play_stop = QToolButton()
        self.btn_play_stop.setText("⏹")
        self.btn_play_stop.setToolTip("停止并恢复显示全部图层")
        for b in (self.btn_play_first, self.btn_play_prev,
                  self.btn_play_next, self.btn_play_stop):
            b.setFixedSize(30, 26)
            pr1.addWidget(b)
        pr1.addWidget(self.btn_play)
        self.scr_play = QSlider(Qt.Horizontal)
        self.scr_play.setRange(0, 0)
        self.scr_play.setToolTip("拖动查看指定帧（自动暂停）")
        pr1.addWidget(self.scr_play, 1)
        self.lbl_play_pos = QLabel("0/0")
        self.lbl_play_pos.setFixedWidth(48)
        self.lbl_play_pos.setAlignment(Qt.AlignCenter)
        self.lbl_play_pos.setStyleSheet("color:#888;")
        pr1.addWidget(self.lbl_play_pos)
        self.cmb_fps = QComboBox()
        self.cmb_fps.addItems(["4 fps", "6 fps", "8 fps", "12 fps",
                               "15 fps", "24 fps"])
        self.cmb_fps.setCurrentText("12 fps")
        self.cmb_fps.setToolTip("播放帧率（帧/秒）")
        pr1.addWidget(QLabel("速度"))
        pr1.addWidget(self.cmb_fps)
        pl.addLayout(pr1)

        pr2 = QHBoxLayout()
        pr2.setSpacing(6)
        self.chk_loop = QCheckBox("循环")
        self.chk_loop.setChecked(True)
        self.btn_jitter = QPushButton("📊 抖动扫描")
        self.btn_jitter.setToolTip(
            "计算所有相邻帧主体中心的位移，列出抖动最大的帧")
        pr2.addWidget(self.chk_loop)
        pr2.addWidget(self.btn_jitter)
        pr2.addSpacing(10)
        pr2.addWidget(QLabel("自动对齐："))
        self.cmb_align = QComboBox()
        self.cmb_align.addItems(["中间对齐", "底部对齐"])
        self.cmb_align.setToolTip(
            "中间对齐：每帧主体包围盒中心对齐到画布中心\n"
            "底部对齐：每帧主体底边对齐到距画布底边指定像素处")
        pr2.addWidget(self.cmb_align)
        self.spn_bottom = QSpinBox()
        self.spn_bottom.setRange(0, 100000)
        self.spn_bottom.setValue(20)
        self.spn_bottom.setSuffix(" px")
        self.spn_bottom.setFixedWidth(80)
        self.spn_bottom.setEnabled(False)   # 默认中间对齐，底边参数不可用
        self.spn_bottom.setToolTip("主体底边距画布底边的像素数")
        pr2.addWidget(self.spn_bottom)
        self.chk_align_cx = QCheckBox("水平居中")
        self.chk_align_cx.setChecked(True)
        self.chk_align_cx.setEnabled(False)
        self.chk_align_cx.setToolTip("底部对齐时，是否同时把每帧主体水平居中")
        pr2.addWidget(self.chk_align_cx)
        self.btn_align = QPushButton("一键对齐")
        self.btn_align.setToolTip(
            "按所选方式移动所有可见帧的主体（各帧可单独撤销）")
        pr2.addWidget(self.btn_align)
        pr2.addStretch()
        self.lbl_jitter = QLabel("")
        self.lbl_jitter.setStyleSheet("color:#2ecc71;font-size:12px;")
        pr2.addWidget(self.lbl_jitter)
        pl.addLayout(pr2)

        self.btn_play_first.clicked.connect(self._play_first)
        self.btn_play_prev.clicked.connect(lambda: self._play_step(-1))
        self.btn_play.clicked.connect(self._play_toggle)
        self.btn_play_next.clicked.connect(lambda: self._play_step(1))
        self.btn_play_stop.clicked.connect(self._play_stop)
        self.scr_play.valueChanged.connect(self._on_play_scrub)
        self.cmb_fps.currentTextChanged.connect(self._on_fps_changed)
        self.cmb_align.currentIndexChanged.connect(self._on_align_mode_changed)
        self.btn_align.clicked.connect(self._auto_align)
        self.btn_jitter.clicked.connect(self._jitter_scan)
        lv.addWidget(play_box, 0)

        self.info_label = QLabel("尚未加载图片")
        self.info_label.setStyleSheet("color:#f1c40f;font-size:12px;")
        lv.addWidget(self.info_label)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;font-size:12px;")
        lv.addWidget(self.status)
        root.addWidget(left_box, 13)

        # ===== 右侧：上参数 + 下图片列 =====
        right = QVBoxLayout()
        right.setSpacing(8)

        # ---- 上半：画布与工具 ----
        tool_box = QGroupBox("画布与工具")
        tf = QVBoxLayout(tool_box)
        tf.setSpacing(5)
        rowt = QHBoxLayout()
        rowt.addWidget(QLabel("工具："))
        self.btn_tool_move = QToolButton()
        self.btn_tool_move.setText("🖐 移动")
        self.btn_tool_move.setCheckable(True)
        self.btn_tool_move.setChecked(True)
        self.btn_tool_move.setToolTip("选中图片后，在画布上拖拽移动位置")
        self.btn_tool_scale = QToolButton()
        self.btn_tool_scale.setText("⤢ 缩放")
        self.btn_tool_scale.setCheckable(True)
        self.btn_tool_scale.setToolTip("选中图片后，在画布上左右拖拽调整大小")
        self.btn_tool_marquee = QToolButton()
        self.btn_tool_marquee.setText("☐ 框选")
        self.btn_tool_marquee.setCheckable(True)
        self.btn_tool_marquee.setToolTip(
            "框选多个图层（Shift 加选），切回移动/缩放可对整组操作")
        from PySide6.QtWidgets import QButtonGroup
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_group.addButton(self.btn_tool_move)
        self.tool_group.addButton(self.btn_tool_scale)
        self.tool_group.addButton(self.btn_tool_marquee)
        self.btn_tool_move.clicked.connect(
            lambda: self.view.set_tool("move"))
        self.btn_tool_scale.clicked.connect(
            lambda: self.view.set_tool("scale"))
        self.btn_tool_marquee.clicked.connect(
            lambda: self.view.set_tool("marquee"))
        rowt.addWidget(self.btn_tool_move)
        rowt.addWidget(self.btn_tool_scale)
        rowt.addWidget(self.btn_tool_marquee)
        btn_zero = QPushButton("位置归零")
        btn_zero.setToolTip("选中图片回到画布中心（可撤销）")
        btn_zero.clicked.connect(self._reset_selected_offset)
        rowt.addWidget(btn_zero)
        self.btn_undo_sel = QPushButton("↩ 撤销本图")
        self.btn_undo_sel.setToolTip(
            "撤销选中图片的上一次移动/缩放（Ctrl+Z，各图独立）")
        self.btn_undo_sel.clicked.connect(self._undo_selected)
        self.btn_undo_sel.setEnabled(False)
        rowt.addWidget(self.btn_undo_sel)
        rowt.addStretch()
        tf.addLayout(rowt)

        # 视图缩放：滚轮以鼠标为锚缩放，中键拖拽平移，双击复位
        rowz = QHBoxLayout()
        rowz.addWidget(QLabel("视图："))
        btn_zoom_out = QToolButton()
        btn_zoom_out.setText("➖")
        btn_zoom_out.setToolTip("缩小预览（滚轮↓）")
        btn_zoom_out.clicked.connect(lambda: self.view.zoom_step(0.8))
        rowz.addWidget(btn_zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(46)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        rowz.addWidget(self.zoom_label)
        btn_zoom_in = QToolButton()
        btn_zoom_in.setText("➕")
        btn_zoom_in.setToolTip("放大预览（滚轮↑）")
        btn_zoom_in.clicked.connect(lambda: self.view.zoom_step(1.25))
        rowz.addWidget(btn_zoom_in)
        btn_fit = QPushButton("适配窗口")
        btn_fit.setToolTip("复位缩放与平移（双击画布同效）")
        btn_fit.clicked.connect(self.view.zoom_reset)
        rowz.addWidget(btn_fit)
        rowz.addWidget(QLabel("｜滚轮缩放 / 中键平移"))
        rowz.addStretch()
        self.view.zoom_changed = lambda pct: \
            self.zoom_label.setText(f"{pct}%")
        tf.addLayout(rowz)

        # 画布宽高作为内部数据存储（隐藏），由「画布大小」对话框驱动；
        # 持久化与 _canvas_size 仍依赖这三个控件
        self.chk_canvas_auto = QCheckBox()
        self.chk_canvas_auto.setChecked(True)
        self.spn_cw = QSpinBox()
        self.spn_cw.setRange(16, 16384)
        self.spn_cw.setValue(1024)
        self.spn_ch = QSpinBox()
        self.spn_ch.setRange(16, 16384)
        self.spn_ch.setValue(1024)
        for _w in (self.chk_canvas_auto, self.spn_cw, self.spn_ch):
            _w.setVisible(False)
            tf.addWidget(_w)
        self.chk_canvas_auto.toggled.connect(self._on_setting_changed)
        self.spn_cw.valueChanged.connect(self._on_setting_changed)
        self.spn_ch.valueChanged.connect(self._on_setting_changed)

        # PS 风格「画布大小 / 图像大小」对话框入口
        rowc = QHBoxLayout()
        self.btn_canvas_size = QPushButton("画布大小…")
        self.btn_canvas_size.setToolTip(
            "调整画板宽高（图像内容不动），支持相对与九宫格定位")
        self.btn_canvas_size.clicked.connect(self._open_canvas_size)
        rowc.addWidget(self.btn_canvas_size)
        self.btn_image_size = QPushButton("图像大小…")
        self.btn_image_size.setToolTip(
            "整幅画布与所有图层一起按比例缩放（各图可单独撤销）")
        self.btn_image_size.clicked.connect(self._open_image_size)
        rowc.addWidget(self.btn_image_size)
        self.canvas_label = QLabel("画布尺寸：-")
        self.canvas_label.setStyleSheet("color:#2ecc71;")
        rowc.addWidget(self.canvas_label)
        rowc.addStretch()
        tf.addLayout(rowc)
        right.addWidget(tool_box, 0)

        # 精灵图参数
        sprite_box = QGroupBox("精灵图 PNG")
        sf = QVBoxLayout(sprite_box)
        sf.setSpacing(5)
        row1 = QHBoxLayout()
        self.chk_auto_cols = QCheckBox("自动行数（近方形）")
        self.chk_auto_cols.setChecked(False)
        self.chk_auto_cols.toggled.connect(self._on_setting_changed)
        row1.addWidget(self.chk_auto_cols)
        row1.addStretch()
        self.spn_rows = QSpinBox()
        self.spn_rows.setRange(1, 999)
        self.spn_rows.setValue(1)
        self.spn_rows.valueChanged.connect(self._on_setting_changed)
        self.chk_auto_cols.toggled.connect(
            lambda on: self.spn_rows.setEnabled(not on))
        row1.addWidget(QLabel("行数"))
        row1.addWidget(self.spn_rows)
        row1.addWidget(QLabel("间距"))
        self.spn_pad = QSpinBox()
        self.spn_pad.setRange(0, 500)
        self.spn_pad.setValue(0)
        self.spn_pad.setSuffix(" px")
        self.spn_pad.valueChanged.connect(self._on_setting_changed)
        row1.addWidget(self.spn_pad)
        sf.addLayout(row1)

        row3 = QHBoxLayout()
        self.size_label = QLabel("输出尺寸：-")
        self.size_label.setStyleSheet("color:#2ecc71;")
        row3.addWidget(self.size_label, 1)
        self.chk_sprite_preview = QCheckBox("预览精灵图")
        self.chk_sprite_preview.toggled.connect(
            lambda *_: self.view.zoom_reset())
        self.chk_sprite_preview.toggled.connect(self._render_preview)
        self.chk_sprite_preview.toggled.connect(
            self._on_sprite_preview_toggled)
        row3.addWidget(self.chk_sprite_preview)
        sf.addLayout(row3)

        row4 = QHBoxLayout()
        self.btn_export_sprite = QPushButton("💾 导出精灵图 PNG")
        self.btn_export_sprite.clicked.connect(self._export_sprite)
        row4.addWidget(self.btn_export_sprite)
        btn_open = QPushButton("📂 打开导出目录")
        btn_open.clicked.connect(self._open_export_dir)
        row4.addWidget(btn_open)
        sf.addLayout(row4)
        right.addWidget(sprite_box, 0)

        # ---- 下半：所有图片（单列，眼睛+缩略图+缩放） ----
        list_box = QGroupBox("所有图片（👁=是否显示，点击行选中）")
        gl = QVBoxLayout(list_box)
        gl.setContentsMargins(4, 6, 4, 4)
        # 标题行：小眼睛总开关（全部显示/隐藏）
        head = QHBoxLayout()
        self.eye_all = QToolButton()
        self.eye_all.setText("👁")
        self.eye_all.setFixedSize(28, 28)
        self.eye_all.setCursor(Qt.PointingHandCursor)
        self.eye_all.setToolTip("全部显示 / 全部隐藏")
        self.eye_all.setStyleSheet(
            "QToolButton{background:transparent;border:none;font-size:15px;}"
            "QToolButton:hover{background:#3a3f4b;border-radius:4px;}")
        self.eye_all.clicked.connect(self._toggle_all_visible)
        head.addWidget(self.eye_all)
        head.addWidget(QLabel("图层列表"))
        head.addStretch()
        self.btn_import = QPushButton("📥 导入图片…")
        self.btn_import.setToolTip(
            "把外部 PNG 图片复制进当前批次，作为新图层参与工作台与导出")
        self.btn_import.clicked.connect(self._import_images)
        head.addWidget(self.btn_import)
        gl.addLayout(head)
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
        right.addWidget(list_box, 1)

        root.addLayout(right, 8)

    # ---------- 导入图片 ----------

    def _import_images(self):
        """把外部 PNG 复制进当前批次的 matted 目录，作为新图层参与导出"""
        if not self.matted_dir or not os.path.isdir(self.matted_dir):
            self.status_text("请先完成批量抠图，建立当前批次后再导入")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要导入的 PNG 图片", "",
            "PNG 图片 (*.png)")
        if not paths:
            return
        imported, skipped, failed = [], [], []
        existing = set(os.listdir(self.matted_dir))
        for path in paths:
            fn = os.path.basename(path)
            if not fn.lower().endswith(".png"):
                skipped.append(fn)
                continue
            # 与现有文件同名时自动改名，避免覆盖批次原图
            name, ext = os.path.splitext(fn)
            cand = fn
            n = 1
            while cand in existing:
                cand = f"{name}_{n}{ext}"
                n += 1
            if eu.load_bgra(path) is None:
                failed.append(fn)
                continue
            try:
                shutil.copy2(path, os.path.join(self.matted_dir, cand))
                existing.add(cand)
                imported.append(cand)
            except OSError:
                failed.append(fn)
        if imported:
            # 批次签名变化 → 触发整体重载，新图层自动带独立变换/撤销
            self._reload_if_changed()
            self.status_text(
                f"已导入 {len(imported)} 张：{'、'.join(imported)}"
                + (f"；跳过非 PNG {len(skipped)} 张" if skipped else "")
                + (f"；读取失败 {len(failed)} 张" if failed else ""))
        else:
            self.status_text(
                "没有可导入的 PNG"
                + (f"（跳过非 PNG {len(skipped)} 张）" if skipped else "")
                + (f"（读取失败 {len(failed)} 张）" if failed else ""))

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
        # 重新加载（含批次变化）：变换撤销历史只保留在内存中，整体丢弃
        self._undo = {}
        self._undo_coalesce = {}
        self._drag_pending = None
        self._ws_cache = None
        self._play_timer.stop()
        self._play_key = None
        self.btn_play.setText("▶ 播放")
        self.lbl_jitter.setText("")
        self.selected_keys = set()
        self._canvas_anchor = (1, 1)       # 画布扩展定位 (row,col) 0..2
        self._canvas_ext_color = "透明"
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
        self._update_playbar()
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
        self._undo = {}
        self._undo_coalesce = {}
        self._drag_pending = None
        self._ws_cache = None
        self._play_timer.stop()
        self._play_key = None
        self.selected_keys = set()
        self._update_undo_ui()
        self._update_playbar()
        self.view.set_image(None)
        self.info_label.setText(text)
        self.size_label.setText("输出尺寸：-")
        self.canvas_label.setText("画布尺寸：-")
        self.btn_export_sprite.setEnabled(False)
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
        self._ws_cache = None
        self.loader = _PreviewLoader(self.items, self.PREVIEW_CACHE)
        self.loader.loaded_one.connect(self._on_preview_one)
        self.loader.done.connect(self._render_preview)
        self.loader.done.connect(
            lambda: self.status_text(
                f"预览就绪：{len(self.items)} 张（来源：{self._stage_name()}）"))
        self.loader.start()
        # 导出读的是全分辨率原文件，不依赖预览缓存，立即可用
        self.btn_export_sprite.setEnabled(True)

    def _on_preview_one(self, key: str, img: np.ndarray, factor: float):
        self.preview_cache[key] = (img, factor)
        # 非透明像素包围盒（换算回全分辨率，右/下开区间），供框选命中
        a = img[:, :, 3]
        ys, xs = np.nonzero(a)
        if len(xs):
            inv = 1.0 / factor if factor > 0 else 1.0
            self._alpha_bbox[key] = (int(xs.min() * inv),
                                     int(ys.min() * inv),
                                     int(np.ceil((xs.max() + 1) * inv)),
                                     int(np.ceil((ys.max() + 1) * inv)))
        for r in self.layer_rows:
            if r.key == key:
                r.set_thumbnail(img)
                break
        if len(self.preview_cache) == 1 or len(self.preview_cache) % 5 == 0:
            self._render_preview()
        self.status_text(f"正在生成预览 {len(self.preview_cache)}"
                         f"/{len(self.items)}")

    # ---------- 设置 ----------

    def _canvas_size(self) -> Tuple[int, int]:
        """画板尺寸：自动=最大帧，否则取手动宽高"""
        if not self.items:
            return (self.spn_cw.value(), self.spn_ch.value())
        if self.chk_canvas_auto.isChecked():
            return (max(it["w"] for it in self.items),
                    max(it["h"] for it in self.items))
        return (self.spn_cw.value(), self.spn_ch.value())

    # ---------- 每图独立撤销（移动/缩放变换） ----------

    UNDO_LIMIT = 50

    def _layer_transform(self, key: str) -> Tuple[float, float, float]:
        pct = float(self._layer_scales.get(key, 100.0))
        ox, oy = self._layer_offsets.get(key, (0.0, 0.0))
        return pct, float(ox), float(oy)

    def _push_undo(self, key: str, mark_coalesce: bool = True):
        """把该图【当前】变换状态作为撤销点入栈（在改动之前调用）"""
        snap = self._layer_transform(key)
        st = self._undo.setdefault(key, [])
        if st and st[-1] == snap:
            return                      # 与栈顶相同，不入重复点
        st.append(snap)
        if len(st) > self.UNDO_LIMIT:
            del st[:-self.UNDO_LIMIT]
        if mark_coalesce:
            self._undo_coalesce[key] = time.monotonic()
        self._update_undo_ui()

    def _undo_layer(self, key: Optional[str]):
        if not key:
            return
        st = self._undo.get(key)
        if not st:
            return
        pct, ox, oy = st.pop()
        if not st:
            self._undo.pop(key, None)      # 栈空后移除，保持字典干净
        self._layer_scales[key] = pct
        if ox == 0.0 and oy == 0.0:
            self._layer_offsets.pop(key, None)
        else:
            self._layer_offsets[key] = [ox, oy]
        row = next((r for r in self.layer_rows if r.key == key), None)
        if row is not None:
            disp = max(row.spin.minimum(),
                       min(row.spin.maximum(), int(round(pct))))
            row.spin.blockSignals(True)
            row.spin.setValue(disp)
            row.spin.blockSignals(False)
        self._undo_coalesce[key] = time.monotonic()
        self._ws_cache = None
        self._update_undo_ui()
        self._schedule_preview()
        self._save_timer.start()
        self.status_text(
            f"{key} 已撤销上一步移动/缩放（本图剩 {len(st)} 步）")

    def _undo_selected(self):
        self._undo_layer(self.selected_key)

    def _update_undo_ui(self):
        for r in self.layer_rows:
            r.set_undo_count(len(self._undo.get(r.key, ())))
        n = len(self._undo.get(self.selected_key, ())) \
            if self.selected_key else 0
        self.btn_undo_sel.setEnabled(n > 0)
        self.btn_undo_sel.setText(f"↩ 撤销本图（{n}）" if n else "↩ 撤销本图")

    # ---------- 框选多选 ----------

    def _marquee_hit(self, rect, it) -> bool:
        """图层非透明内容在画布上的实际位置是否与框选矩形相交"""
        key = it["key"]
        # 优先用非透明像素包围盒；尚无（未加载）时回退整帧
        bbox = self._alpha_bbox.get(key)
        if bbox is None:
            bbox = (0, 0, it["w"], it["h"])
        l, t, r, b = bbox
        ox, oy = self._layer_offsets.get(key, (0.0, 0.0))
        pf = self._workspace_preview_factor(*self._canvas_size())
        cw, ch = self._canvas_size()
        pw, ph = int(cw * pf), int(ch * pf)
        k = self._layer_scales.get(key, 100) / 100.0
        # 图层缩放后仍按整帧中心居中，包围盒随缩放一起平移
        fw, fh = it["w"] * k * pf, it["h"] * k * pf
        fx, fy = eu.centered_xy(pw, ph, fw, fh)
        fx += ox * pf
        fy += oy * pf
        x0 = fx + l * k * pf
        y0 = fy + t * k * pf
        x1 = fx + r * k * pf
        y1 = fy + b * k * pf
        rx0, ry0 = rect.x(), rect.y()
        rx1, ry1 = rx0 + rect.width(), ry0 + rect.height()
        return not (x0 >= rx1 or x1 <= rx0 or y0 >= ry1 or y1 <= ry0)

    def _marquee_keys(self, rect) -> List[str]:
        return [it["key"] for it in self.items
                if self._layer_visible.get(it["key"], True)
                and self._marquee_hit(rect, it)]

    def _on_marquee_changed(self, rect, additive: bool):
        hits = self._marquee_keys(rect)
        if additive:
            hits = sorted(set(self.selected_keys) | set(hits))
        self._apply_marquee_selection(hits, transient=True)

    def _on_marquee_done(self, rect, additive: bool):
        hits = self._marquee_keys(rect)
        if additive:
            hits = sorted(set(self.selected_keys) | set(hits))
        self._apply_marquee_selection(hits, transient=False)

    def _apply_marquee_selection(self, keys: List[str], transient: bool):
        self.selected_keys = set(keys)
        for r in self.layer_rows:
            r.set_marqueed(r.key in self.selected_keys)
        n = len(self.selected_keys)
        if n:
            # 若主选不在组内，把组内第一个提升为主选（供撤销/缩放参考）
            if self.selected_key not in self.selected_keys:
                self.selected_key = sorted(self.selected_keys)[0]
                self._sync_row_selection()
                self._update_undo_ui()
            self.info_label.setText(
                f"已框选 {n} 个图层"
                + ("（预览）" if transient else
                   "，切到移动/缩放工具可整组操作"))
        elif not transient:
            self.info_label.setText("框选为空")
        if not transient:
            self._render_workspace()

    def _clear_marquee(self):
        if self.selected_keys:
            self.selected_keys = set()
            for r in self.layer_rows:
                r.set_marqueed(False)
            self._render_workspace()

    # ---------- 画布拖拽交互 ----------

    def _on_drag_started(self):
        # 首个实际变化时才入栈，纯点击不留无效撤销点
        if self.selected_key and not self.chk_sprite_preview.isChecked():
            self._drag_pending = self.selected_key

    def _on_drag_finished(self):
        self._drag_pending = None
        self._save_timer.start()

    def _on_view_resized(self):
        if self.items:
            self._ws_cache = None      # 预览分辨率随控件变化，缓存失效
            self._render_preview()

    def _drag_keys(self) -> List[str]:
        """本次拖拽作用的目标图层：主选优先；若在框选组内则整组"""
        key = self.selected_key
        if not key:
            return []
        if self.selected_keys and key in self.selected_keys:
            # 仅可见图层参与成组操作，且组内至少要有当前主选
            return sorted(k for k in self.selected_keys
                          if self._layer_visible.get(k, True))
        return [key]

    def _on_canvas_drag(self, dx: float, dy: float):
        """移动工具=拖动位置；缩放工具=横向拖拽调大小（支持整组）"""
        keys = self._drag_keys()
        if not keys:
            return
        key = self.selected_key
        if self._drag_pending == key:
            for k in keys:
                self._push_undo(k)     # 拖拽事务首个变化：整组各入栈一次
            self._drag_pending = None
        if self.view.tool == "move":
            for k in keys:
                ox, oy = self._layer_offsets.get(k, (0.0, 0.0))
                self._layer_offsets[k] = [ox + dx, oy + dy]
            # 拖拽中静态层命中缓存，单帧仅 ~3ms，立即跟手渲染
            self._render_preview()
        else:
            for k in keys:
                row = next((r for r in self.layer_rows
                            if r.key == k), None)
                if row is not None:
                    cur = row.spin.value()
                    new = max(5, min(400, int(round(cur + dx * 0.5))))
                    if new != cur:
                        row.spin.setValue(new)  # 触发 changed → 预览+保存

    # ---------- PS 风格「画布大小 / 图像大小」 ----------

    def _open_canvas_size(self):
        """画布大小：改画板宽高，图像内容不动，九宫格定位决定留白方向"""
        if not self.items:
            self.status_text("当前没有批次，无法调整画布")
            return
        cw, ch = self._canvas_size()
        dlg = CanvasSizeDialog(self, cw, ch, cw, ch,
                               self._canvas_anchor,
                               self._canvas_ext_color)
        if dlg.exec() != QDialog.Accepted:
            return
        new_w, new_h, anchor, color = dlg.result_values()
        self._canvas_anchor = anchor
        self._canvas_ext_color = color
        # 切为手动并写入新宽高（抑制中途信号，统一在末尾刷新一次）
        self.chk_canvas_auto.blockSignals(True)
        self.spn_cw.blockSignals(True)
        self.spn_ch.blockSignals(True)
        self.chk_canvas_auto.setChecked(False)
        self.spn_cw.setValue(new_w)
        self.spn_ch.setValue(new_h)
        self.chk_canvas_auto.blockSignals(False)
        self.spn_cw.blockSignals(False)
        self.spn_ch.blockSignals(False)
        self._apply_canvas_anchor(cw, ch, new_w, new_h, anchor)
        # 显式同步画布尺寸/输出尺寸标签（setValue 信号已被抑制）
        self._update_size_info()
        self.status_text(
            f"画布大小 {cw}×{ch} → {new_w}×{new_h}"
            f"（锚点 {['左上','上','右上','左','中','右','左下','下','右下'][anchor[0]*3+anchor[1]]}，"
            f"扩展颜色 {color}）")

    def _apply_canvas_anchor(self, old_w: int, old_h: int,
                             new_w: int, new_h: int,
                             anchor: Tuple[int, int]):
        """按九宫格锚点平移所有图层，使图像内容在新画布中位置正确"""
        dx = new_w - old_w
        dy = new_h - old_h
        if dx == 0 and dy == 0:
            return
        ar, ac = anchor
        # 锚点决定内容偏移：中=(0,0)，左上=(-dx/2,-dy/2)，右下=(dx/2,dy/2)
        off_x = dx * (ac - 1) / 2.0
        off_y = dy * (ar - 1) / 2.0
        if off_x == 0 and off_y == 0:
            return
        # 所有图层（含从未移动过、不在 _layer_offsets 里的）都要跟随锚点
        for it in self.items:
            key = it["key"]
            self._push_undo(key)
            ox, oy = self._layer_offsets.get(key, (0.0, 0.0))
            self._layer_offsets[key] = [ox + off_x, oy + off_y]
        self._ws_cache = None
        self._schedule_preview()
        self._save_timer.start()

    def _open_image_size(self):
        """图像大小：整幅画布+所有图层一起按比例缩放"""
        if not self.items:
            self.status_text("当前没有批次，无法调整图像大小")
            return
        cw, ch = self._canvas_size()
        dlg = ImageSizeDialog(self, cw, ch)
        if dlg.exec() != QDialog.Accepted:
            return
        scale = dlg.result_scale()
        if scale <= 0 or abs(scale - 1.0) < 1e-6:
            return
        # 画布同步缩放（切为手动；抑制信号避免 setValue 触发
        # _on_setting_changed 干扰，统一在末尾刷新）
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        self.chk_canvas_auto.blockSignals(True)
        self.spn_cw.blockSignals(True)
        self.spn_ch.blockSignals(True)
        self.chk_canvas_auto.setChecked(False)
        self.spn_cw.setValue(new_w)
        self.spn_ch.setValue(new_h)
        self.chk_canvas_auto.blockSignals(False)
        self.spn_cw.blockSignals(False)
        self.spn_ch.blockSignals(False)
        # 所有图层（含从未动过的）缩放与偏移同步按比例变化（各入一个撤销点）
        for it in self.items:
            key = it["key"]
            self._push_undo(key)
            cur = self._layer_scales.get(key, 100)
            self._layer_scales[key] = max(0.01, float(cur) * scale)
            ox, oy = self._layer_offsets.get(key, (0.0, 0.0))
            if ox or oy:
                self._layer_offsets[key] = [ox * scale, oy * scale]
        self._ws_cache = None
        self._sync_row_scales()
        self._update_size_info()
        self._render_preview()
        self._save_timer.start()
        self.status_text(
            f"图像大小已整体缩放 {scale*100:.1f}%"
            f"（画布 {new_w}×{new_h}，各图可单独撤销）")

    def _reset_selected_offset(self):
        key = self.selected_key
        if key and key in self._layer_offsets:
            self._push_undo(key)
            del self._layer_offsets[key]
            self._ws_cache = None
            self._schedule_preview()
            self._save_timer.start()
            self.status_text(f"{key} 已回到画布中心（可撤销）")

    def _sprite_params(self) -> dict:
        return {
            "auto_columns": self.chk_auto_cols.isChecked(),
            # 用户填的是行数；换算成列数传给拼接逻辑（列数=ceil(n/行数)）
            "rows": self.spn_rows.value(),
            "padding": self.spn_pad.value(),
        }

    def _cols_from_rows(self, rows: int, n: int) -> int:
        """按目标行数反推列数：ceil(n/rows)，至少 1 列"""
        if n <= 0:
            return 0
        return max(1, int(np.ceil(n / max(1, rows))))

    def _on_setting_changed(self):
        self._update_size_info()
        self._schedule_preview()
        self._save_timer.start()

    def _schedule_preview(self, *a):
        if self.preview_cache:
            self._preview_timer.start()

    def _visible_items(self) -> List[dict]:
        return [it for it in self.items
                if self._layer_visible.get(it["key"], True)]

    # ---------- 帧播放（检查抖动） ----------

    def _play_visible_keys(self) -> List[str]:
        return [it["key"] for it in self._visible_items()]

    def _play_fps(self) -> int:
        try:
            return max(1, int(self.cmb_fps.currentText().split()[0]))
        except (ValueError, IndexError):
            return 12

    def _update_playbar(self):
        """根据当前可见帧列表同步滑块范围/标签；播放帧失效时修正"""
        keys = self._play_visible_keys()
        n = len(keys)
        on = n > 0
        for w in (self.btn_play_first, self.btn_play_prev, self.btn_play,
                  self.btn_play_next, self.btn_play_stop, self.scr_play,
                  self.cmb_fps, self.btn_jitter, self.chk_loop,
                  self.cmb_align, self.btn_align):
            w.setEnabled(on)
        self.spn_bottom.setEnabled(on and self.cmb_align.currentIndex() == 1)
        self.chk_align_cx.setEnabled(
            on and self.cmb_align.currentIndex() == 1)
        need_render = False
        if self._play_key is not None and self._play_key not in keys:
            self._play_timer.stop()
            self.btn_play.setText("▶ 播放")
            self._play_key = keys[0] if n else None
            need_render = n > 0
        v = (keys.index(self._play_key) + 1) if self._play_key in keys else 1
        self.scr_play.blockSignals(True)
        self.scr_play.setRange(1, max(1, n))
        self.scr_play.setValue(v)
        self.scr_play.blockSignals(False)
        self.lbl_play_pos.setText(f"{v if n else 0}/{n}")
        if not n:
            self.lbl_jitter.setText("")
        if need_render:
            self._ws_cache = None
            self._schedule_preview()
        elif self._play_key is not None:
            self._update_jitter_label()

    def _play_alpha_map(self) -> Optional[Dict[str, float]]:
        """播放模式下只显示当前帧"""
        if not self._play_key:
            return None
        keys = self._play_visible_keys()
        if self._play_key not in keys:
            return None
        return {self._play_key: 1.0}

    def _play_toggle(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶ 播放")
            self._ws_cache = None
            self._render_preview()
            return
        keys = self._play_visible_keys()
        if not keys:
            self.status_text("没有可见帧可播放")
            return
        if self.chk_sprite_preview.isChecked():
            self.chk_sprite_preview.setChecked(False)  # 切回工作台
        if self._play_key not in keys:
            self._play_key = keys[0]
        self._play_timer.start(max(1, int(1000 / self._play_fps())))
        self.btn_play.setText("⏸ 暂停")
        self._ws_cache = None
        self._render_preview()

    def _play_tick(self):
        """定时器：前进一帧，循环或停在末尾"""
        keys = self._play_visible_keys()
        if not keys:
            self._play_stop()
            return
        i = keys.index(self._play_key) if self._play_key in keys else -1
        j = i + 1
        if j >= len(keys):
            if self.chk_loop.isChecked():
                j = 0
            else:
                self._play_timer.stop()
                self.btn_play.setText("▶ 播放")
                return
        self._play_seek(keys[j])

    def _play_step(self, delta: int):
        """按钮：相对步进（自动暂停）"""
        keys = self._play_visible_keys()
        if not keys:
            return
        self._play_timer.stop()
        self.btn_play.setText("▶ 播放")
        if self.chk_sprite_preview.isChecked():
            self.chk_sprite_preview.setChecked(False)
        i = keys.index(self._play_key) if self._play_key in keys else (
            0 if delta > 0 else len(keys) - 1)
        j = min(max(0, i + delta), len(keys) - 1)
        self._play_seek(keys[j])

    def _play_first(self):
        keys = self._play_visible_keys()
        if not keys:
            return
        self._play_timer.stop()
        self.btn_play.setText("▶ 播放")
        if self.chk_sprite_preview.isChecked():
            self.chk_sprite_preview.setChecked(False)
        self._play_seek(keys[0])

    def _play_seek(self, key: str):
        self._play_key = key
        keys = self._play_visible_keys()
        v = keys.index(key) + 1 if key in keys else 1
        self.scr_play.blockSignals(True)
        self.scr_play.setValue(v)
        self.scr_play.blockSignals(False)
        self.lbl_play_pos.setText(f"{v}/{len(keys)}")
        self._ws_cache = None
        self._render_preview()
        self._update_jitter_label()

    def _play_stop(self):
        self._play_timer.stop()
        self.btn_play.setText("▶ 播放")
        if self._play_key is None:
            return
        self._play_key = None
        self.lbl_jitter.setText("")
        self._ws_cache = None
        self._render_preview()

    def _on_play_scrub(self, v: int):
        keys = self._play_visible_keys()
        if not keys:
            return
        self._play_timer.stop()
        self.btn_play.setText("▶ 播放")
        if self.chk_sprite_preview.isChecked():
            self.chk_sprite_preview.setChecked(False)
        j = min(max(1, v), len(keys)) - 1
        self._play_seek(keys[j])

    def _on_fps_changed(self):
        if self._play_timer.isActive():
            self._play_timer.setInterval(max(1, int(1000 / self._play_fps())))

    def _on_align_mode_changed(self, idx: int):
        # 中间对齐恒为水平+垂直居中；底边参数与水平居中选项仅底部模式可用
        bottom = idx == 1
        self.spn_bottom.setEnabled(bottom and bool(self.items))
        self.chk_align_cx.setEnabled(bottom and bool(self.items))

    def _subject_bbox_full(self, key: str):
        """从源文件直接读全分辨率主体 alpha 包围盒 (l,t,r,b)，右/下开区间"""
        it = next((x for x in self.items if x["key"] == key), None)
        if it is None:
            return None
        img = imread_unicode(it["src"])
        if img is None or img.ndim < 3 or img.shape[2] < 4:
            return None
        ys, xs = np.nonzero(img[..., 3] > 16)
        if xs.size == 0:
            return None
        return (int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1)

    def _auto_align(self):
        """
        自动对齐所有可见帧（每帧各入一个撤销点）：
          中间对齐：主体包围盒中心 → 画布中心（X/Y）
          底部对齐：主体底边 → 距画布底边 N px；可选同时水平居中
        """
        vis = self._visible_items()
        if len(vis) < 1:
            self.status_text("没有可见帧可对齐")
            return
        mode_bottom = self.cmb_align.currentIndex() == 1
        margin = self.spn_bottom.value()
        center_x = self.chk_align_cx.isChecked()
        cw, ch = self._canvas_size()

        moved = 0
        skipped = []
        max_shift = 0.0
        for it in vis:
            key = it["key"]
            bbox = self._subject_bbox_full(key)
            if bbox is None:
                skipped.append(key)
                continue
            l, t, r, b = bbox
            pct = float(self._layer_scales.get(key, 100)) / 100.0
            w, h = float(it["w"]), float(it["h"])
            cx = (l + r) / 2.0
            by = float(b)
            ox_new, oy_new = self._layer_offsets.get(key, (0.0, 0.0))
            if mode_bottom:
                # 画布中主体底边 = (ch-h*pct)/2 + oy + b*pct，目标 ch-margin
                oy_new = ch - margin - (ch - h * pct) / 2.0 - by * pct
                if center_x:
                    ox_new = pct * (w / 2.0 - cx)
            else:
                # 画布中主体中心 = 画布中心
                ox_new = pct * (w / 2.0 - cx)
                oy_new = pct * (h / 2.0 - (t + b) / 2.0)
            ox_old, oy_old = self._layer_offsets.get(key, (0.0, 0.0))
            shift = float(np.hypot(ox_new - ox_old, oy_new - oy_old))
            if shift > 0.01:
                self._push_undo(key)
                self._layer_offsets[key] = [ox_new, oy_new]
                moved += 1
                max_shift = max(max_shift, shift)
        if not moved:
            self.status_text("所有帧已在对齐位置，无需移动")
            return
        self._ws_cache = None
        self._render_preview()
        self._save_timer.start()
        if self._play_key is not None:
            self._update_jitter_label()
        name = f"底部对齐（距底边 {margin}px" \
               + ("，水平居中" if mode_bottom and center_x else "") + "）" \
            if mode_bottom else "中间对齐"
        msg = f"{name}：已移动 {moved}/{len(vis)} 帧，最大位移 {max_shift:.1f}px"
        if skipped:
            msg += f"，{len(skipped)} 帧无主体跳过"
        self.status_text(msg)

    def _on_sprite_preview_toggled(self, sprite_mode: bool):
        # 切到精灵图预览时暂停帧播放（保留当前帧，切回来可继续）
        if sprite_mode and self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶ 播放")

    def _subject_center(self, key: str):
        """主体 alpha 包围盒中心在当前画布坐标系中的位置（含缩放/偏移）"""
        cached = self.preview_cache.get(key)
        if cached is None:
            return None
        im, cf = cached
        ys, xs = np.where(im[..., 3] > 16)
        if xs.size == 0:
            return None
        bx = (int(xs.min()) + int(xs.max()) + 1) / 2.0 / cf
        by = (int(ys.min()) + int(ys.max()) + 1) / 2.0 / cf
        it = next((x for x in self.items if x["key"] == key), None)
        if it is None:
            return None
        pct = float(self._layer_scales.get(key, 100)) / 100.0
        cw, ch = self._canvas_size()
        ox, oy = self._layer_offsets.get(key, (0.0, 0.0))
        return (cw / 2.0 + ox + (bx - it["w"] / 2.0) * pct,
                ch / 2.0 + oy + (by - it["h"] / 2.0) * pct)

    def _frame_shift(self, idx: int, keys: List[str]):
        """当前帧相对前一可见帧的主体中心位移 (dx, dy, dist)"""
        if idx <= 0 or idx >= len(keys):
            return None
        c0 = self._subject_center(keys[idx - 1])
        c1 = self._subject_center(keys[idx])
        if c0 is None or c1 is None:
            return None
        dx, dy = c1[0] - c0[0], c1[1] - c0[1]
        return dx, dy, float(np.hypot(dx, dy))

    def _update_jitter_label(self):
        keys = self._play_visible_keys()
        if not self._play_key or self._play_key not in keys:
            self.lbl_jitter.setText("")
            return
        i = keys.index(self._play_key)
        sh = self._frame_shift(i, keys)
        if sh is None:
            self.lbl_jitter.setText("第 1 帧（无前帧对比）")
            self.lbl_jitter.setStyleSheet("color:#888;font-size:12px;")
            return
        dx, dy, d = sh
        warn = d >= 5.0
        self.lbl_jitter.setText(
            f"相对上帧 Δx={dx:+.1f} Δy={dy:+.1f}｜位移 {d:.1f} px"
            + ("  ⚠ 抖动偏大" if warn else ""))
        self.lbl_jitter.setStyleSheet(
            ("color:#e74c3c;" if warn else "color:#2ecc71;")
            + "font-size:12px;")

    def _jitter_scan(self):
        """扫描所有相邻可见帧的主体中心位移，列出抖动最大的帧"""
        keys = self._play_visible_keys()
        if len(keys) < 2:
            QMessageBox.information(self, "抖动扫描", "至少需要 2 张可见帧")
            return
        if any(self.preview_cache.get(k) is None for k in keys):
            QMessageBox.information(
                self, "抖动扫描", "图片仍在加载，请稍候再试")
            return
        num = {it["key"]: it["number"] for it in self.items}
        recs = []
        for i in range(1, len(keys)):
            sh = self._frame_shift(i, keys)
            if sh is not None:
                recs.append((i, keys[i], *sh))
        if not recs:
            QMessageBox.information(self, "抖动扫描", "未能检测到主体像素")
            return
        dists = [r[4] for r in recs]
        avg = float(np.mean(dists))
        top = sorted(recs, key=lambda r: -r[4])[:5]
        lines = [
            f"共 {len(keys)} 张可见帧，平均帧间位移 {avg:.2f} px",
            "位移最大的帧（编号 / Δx / Δy / 距离）：",
            "",
        ]
        for _, key, dx, dy, d in top:
            lines.append(f"  帧 {num.get(key, '?'):>3}："
                         f"Δx={dx:+6.1f}  Δy={dy:+6.1f}  {d:6.1f} px")
        lines += ["", "判定参考：<2 px 很稳；2~5 px 轻微；≥5 px 建议对齐该帧"]
        msg = "\n".join(lines)
        self.status_text(
            f"抖动扫描：平均 {avg:.2f}px，"
            f"最大 {top[0][4]:.1f}px（帧 {num.get(top[0][1], '?')}）")
        QMessageBox.information(self, "抖动扫描结果", msg)

    def _update_size_info(self):
        if not self.items:
            self.size_label.setText("输出尺寸：-")
            self.canvas_label.setText("画布尺寸：-")
            return
        p = self._sprite_params()
        vis = self._visible_items()
        if vis:
            scaled_sizes = [
                (it["w"] * self._layer_scales.get(it["key"], 100) / 100.0,
                 it["h"] * self._layer_scales.get(it["key"], 100) / 100.0)
                for it in vis]
            cols_arg = 0 if p["auto_columns"] else \
                self._cols_from_rows(p["rows"], len(vis))
            W, H, cols, rows = eu.sprite_sheet_size(
                scaled_sizes,
                columns=cols_arg,
                scale=1.0,
                padding=p["padding"])
            self.size_label.setText(
                f"输出尺寸：{W} × {H} px（{cols}列×{rows}行，"
                f"{len(vis)}/{len(self.items)} 帧可见）")
        else:
            self.size_label.setText("输出尺寸：-（全部图层已隐藏）")
        cw, ch = self._canvas_size()
        mode = "自动" if self.chk_canvas_auto.isChecked() else "手动"
        self.canvas_label.setText(f"画布尺寸：{cw} × {ch} px（{mode}）")

    # ---------- 图层列表 ----------

    def _rebuild_layer_list(self):
        while self.layer_layout.count() > 1:
            it = self.layer_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.layer_rows = []
        for it in self.items:
            pct = int(self._layer_scales.get(it["key"], 100))
            vis = bool(self._layer_visible.get(it["key"], True))
            row = LayerRow(it["key"], it["number"], it["w"], it["h"],
                           pct, vis)
            row.clicked.connect(self._select_layer)
            row.scale_changed.connect(self._on_layer_scale_changed)
            row.visibility_changed.connect(self._on_layer_visibility)
            row.undo_clicked.connect(self._undo_layer)
            self.layer_layout.insertWidget(self.layer_layout.count() - 1, row)
            self.layer_rows.append(row)
        self._sync_row_selection()
        self._update_undo_ui()

    def _select_layer(self, key: str):
        self.selected_key = key
        self.chk_sprite_preview.setChecked(False)
        self._clear_marquee()
        self._sync_row_selection()
        self._update_undo_ui()
        self._render_preview()

    def _sync_row_selection(self):
        for r in self.layer_rows:
            r.set_selected(r.key == self.selected_key)

    def _sync_row_scales(self):
        """批量把 _layer_scales 写回各行 spin（抑制信号，显示取整）"""
        for r in self.layer_rows:
            pct = int(round(self._layer_scales.get(r.key, 100)))
            pct = max(r.spin.minimum(), min(r.spin.maximum(), pct))
            if r.spin.value() != pct:
                r.spin.blockSignals(True)
                r.spin.setValue(pct)
                r.spin.blockSignals(False)

    def _on_layer_scale_changed(self, key: str, pct: int):
        # 键盘连按/滚轮/拖拽产生的连续变化合并为一个撤销点
        now = time.monotonic()
        if now - self._undo_coalesce.get(key, 0.0) > 0.8:
            self._push_undo(key)
        self._undo_coalesce[key] = now
        self._layer_scales[key] = pct
        # 选中层缩放只重画它自己（静态缓存命中），立即跟手；
        # 非选中层缩放改变静态签名，走防抖重建
        if key == self.selected_key:
            self._render_preview()
        else:
            self._schedule_preview()
        self._save_timer.start()

    def _on_layer_visibility(self, key: str, visible: bool):
        self._layer_visible[key] = visible
        self._update_size_info()
        self._update_playbar()
        self._schedule_preview()
        self._save_timer.start()

    def _set_all_visible(self, visible: bool):
        for r in self.layer_rows:
            r.set_visible(visible)          # 批量设置不发单行信号
            self._layer_visible[r.key] = visible
        self.eye_all.setText("👁" if visible else "—")
        self._update_size_info()
        self._update_playbar()
        self._schedule_preview()
        self._save_timer.start()
        self.status_text("全部图层已显示" if visible else "全部图层已隐藏")

    def _toggle_all_visible(self):
        # 只要有任何一个可见图层，就视为"显示中"→切换为全部隐藏
        any_visible = any(
            self._layer_visible.get(r.key, True) for r in self.layer_rows)
        self._set_all_visible(not any_visible)

    # ---------- 工作台渲染 ----------

    def _render_preview(self):
        if not self.items:
            self.view.set_image(None)
            return
        if self.chk_sprite_preview.isChecked():
            self._render_sprite_preview()
        else:
            self._render_workspace()

    def _ready_images(self) -> Dict[str, Tuple[np.ndarray, float]]:
        return dict(self.preview_cache)

    def _workspace_preview_factor(self, cw: int, ch: int) -> float:
        """预览分辨率按控件实际显示尺寸走（封顶 PREVIEW_MAX），避免超渲染"""
        vw = max(1, self.view.width())
        vh = max(1, self.view.height())
        target = min(float(self.PREVIEW_MAX), float(max(vw, vh)))
        return min(1.0, target / max(1.0, float(max(cw, ch))))

    def _render_workspace(self):
        """PS 式工作台：合成所有可见图层（含独立缩放与位置偏移）"""
        cache = self._ready_images()
        if not cache or not self.items:
            return
        cw, ch = self._canvas_size()
        self.view.canvas_wh = (cw, ch)
        pf = self._workspace_preview_factor(cw, ch)
        pw = max(1, int(cw * pf))
        ph = max(1, int(ch * pf))
        # 帧播放模式：只显示当前帧，不画选中框
        play_amap = self._play_alpha_map()
        play_mode = play_amap is not None
        sel = None if play_mode else self.selected_key
        sel_idx = -1 if play_mode else next(
            (i for i, it in enumerate(self.items) if it["key"] == sel), -1)

        def place(it: dict, alpha: float = 1.0):
            """缩放后图层 + 它在预览画布上的左上角坐标"""
            im, cf = cache[it["key"]]
            pct = self._layer_scales.get(it["key"], 100) / 100.0
            sim = eu.scale_bgra(im, pct * pf / cf)
            if alpha < 0.999:
                a = sim[..., 3]
                np.multiply(a, alpha, out=a, casting="unsafe")
            x, y = eu.centered_xy(pw, ph, sim.shape[1], sim.shape[0])
            ox, oy = self._layer_offsets.get(it["key"], (0.0, 0.0))
            return sim, x + int(round(ox * pf)), y + int(round(oy * pf))

        # 除选中层外全部是静态层：签名不变（=正在拖拽选中层）时直接复用
        play_sig = (tuple(sorted(play_amap.items()))
                    if play_amap is not None else None)
        static_sig = (
            pw, ph, sel, tuple(sorted(cache.keys())), play_sig,
            tuple((it["key"],
                   self._layer_scales.get(it["key"], 100),
                   round(float(self._layer_offsets.get(
                       it["key"], (0.0, 0.0))[0]), 2),
                   round(float(self._layer_offsets.get(
                       it["key"], (0.0, 0.0))[1]), 2),
                   self._layer_visible.get(it["key"], True))
                  for it in self.items if it["key"] != sel))
        wc = self._ws_cache
        if wc is None or wc.get("sig") != static_sig:
            # 静态层分两条链（均为 flatten_over 同口径缓冲）：
            # below = 选中层以下；above = 选中层以上。二者缓存为 QPixmap，
            # 拖拽帧只重画选中层，最终 over 合成交给 QPainter。
            below = np.zeros((ph, pw, 4), np.uint8)
            above = None
            shown = 0
            for i, it in enumerate(self.items):
                if it["key"] == sel:
                    continue
                if cache.get(it["key"]) is None:
                    continue
                if play_mode:
                    alpha = play_amap.get(it["key"])
                    if alpha is None:
                        continue
                else:
                    if not self._layer_visible.get(it["key"], True):
                        continue
                    alpha = 1.0
                sim, px, py = place(it, alpha)
                if sel_idx < 0 or i < sel_idx:
                    eu.flatten_over(below, sim, px, py)
                else:
                    if above is None:
                        above = np.zeros((ph, pw, 4), np.uint8)
                    eu.flatten_over(above, sim, px, py)
                shown += 1
            anchor = QPixmap(pw, ph)
            anchor.fill(Qt.transparent)
            wc = {"sig": static_sig,
                  "anchor": anchor,
                  "below_pm": bgra_to_pixmap(below, checker=False),
                  "above_pm": bgra_to_pixmap(above, checker=False)
                  if above is not None else None,
                  "static_shown": shown}
            self._ws_cache = wc

        shown = wc["static_shown"]
        sel_pm = None
        sel_rect = None
        if (not play_mode and sel_idx >= 0 and sel in cache
                and self._layer_visible.get(sel, True)):
            sim, px, py = place(self.items[sel_idx])
            sel_pm = bgra_to_pixmap(sim, checker=False)
            sel_rect = (px, py, sim.shape[1], sim.shape[0])
            shown += 1
        self.view.set_workspace_layers(
            wc["anchor"], wc["below_pm"], wc["above_pm"],
            sel_pm, sel_rect)
        if play_mode:
            keys = self._play_visible_keys()
            i = keys.index(self._play_key) if self._play_key in keys else -1
            icon = "⏸" if self._play_timer.isActive() else "⏯"
            self.info_label.setText(
                f"{icon} 帧播放 {i + 1}/{len(keys)}：{self._play_key}")
            return
        total = len(self.items)
        hidden = total - len(self._visible_items())
        sel = self.selected_key or "-"
        self.info_label.setText(
            f"工作台 {cw}×{ch}：{shown} 个可见图层"
            + (f"，{hidden} 个已隐藏" if hidden else "")
            + f"｜选中：{sel}")

    def _render_sprite_preview(self):
        cache = self._ready_images()
        ready = [(it, cache[it["key"]]) for it in self._visible_items()
                 if it["key"] in cache]
        if not ready:
            self.view.set_image(None)
            self.info_label.setText("没有可见图层可拼接")
            return
        p = self._sprite_params()
        cols_arg = 0 if p["auto_columns"] else \
            self._cols_from_rows(p["rows"], len(ready))
        # 用各帧原始尺寸×各自图层缩放推算全分辨率输出，再整体缩到预览上限
        scaled_sizes = [
            (it["w"] * self._layer_scales.get(it["key"], 100) / 100.0,
             it["h"] * self._layer_scales.get(it["key"], 100) / 100.0)
            for it, _ in ready]
        W0, H0, cols, rows = eu.sprite_sheet_size(
            scaled_sizes,
            columns=cols_arg, scale=1.0, padding=p["padding"])
        target = min(float(self.PREVIEW_MAX),
                     float(max(1, self.view.width(), self.view.height())))
        pf = min(1.0, target / max(1.0, max(W0, H0)))
        sims = []
        for it, (im, cf) in ready:
            layer_scale = self._layer_scales.get(it["key"], 100) / 100.0
            sims.append(eu.scale_bgra(im, layer_scale * pf / cf))
        sheet, _ = eu.build_sprite_sheet(
            sims, columns=cols_arg, scale=1.0,
            padding=max(0, int(round(p["padding"] * pf))))
        self.view.set_image(bgra_to_pixmap(sheet, checker=False))
        self.info_label.setText(
            f"精灵图预览（{len(ready)} 个可见帧，{cols}列×{rows}行，"
            f"输出 {W0}×{H0}，预览缩放 {pf:.2f}×）")

    # ---------- 导出 ----------

    def _export_sprite(self):
        if not self.items or self.worker is not None:
            return
        out_path = os.path.join(self.export_dir, "sprite_sheet.png")
        params = self._sprite_params()
        params["layer_scales"] = dict(self._layer_scales)
        params["layer_visible"] = dict(self._layer_visible)
        params["layer_offsets"] = dict(self._layer_offsets)
        params["canvas_wh"] = self._canvas_size()
        # 换算好列数传给 worker（0=自动近方形；否则按目标行数反推）
        n_vis = sum(1 for it in self.items if self._visible(it))
        params["columns"] = 0 if params["auto_columns"] else \
            self._cols_from_rows(params["rows"], n_vis)
        self.worker = _ExportWorker("sprite", self.items, params, out_path)
        self.worker.progress.connect(self.status_text)
        self.worker.failed.connect(self._on_export_failed)
        self.worker.finished_ok.connect(self._on_export_ok)
        self.worker.finished.connect(self._on_worker_end)
        self.btn_export_sprite.setEnabled(False)
        self.worker.start()

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
        self.chk_auto_cols.setChecked(bool(data.get("auto_columns", False)))
        self.spn_rows.setValue(int(data.get("rows", 1)))
        self.spn_pad.setValue(int(data.get("padding", 0)))
        self.chk_sprite_preview.setChecked(
            bool(data.get("sprite_preview", False)))
        self.chk_canvas_auto.setChecked(bool(data.get("canvas_auto", True)))
        self.spn_cw.setValue(max(16, int(data.get("canvas_w", 1024))))
        self.spn_ch.setValue(max(16, int(data.get("canvas_h", 1024))))
        anc = data.get("canvas_anchor", [1, 1])
        if isinstance(anc, (list, tuple)) and len(anc) == 2:
            self._canvas_anchor = (int(anc[0]), int(anc[1]))
        self._canvas_ext_color = str(data.get("canvas_ext_color", "透明"))
        keys = {it["key"] for it in self.items}
        saved_scales = data.get("layer_scales", {})
        self._layer_scales = {
            k: max(5, min(400, int(v)))
            for k, v in saved_scales.items() if k in keys}
        saved_vis = data.get("layer_visible", {})
        self._layer_visible = {k: bool(v)
                               for k, v in saved_vis.items() if k in keys}
        saved_off = data.get("layer_offsets", {})
        self._layer_offsets = {k: [float(v[0]), float(v[1])]
                               for k, v in saved_off.items()
                               if k in keys and isinstance(v, (list, tuple))
                               and len(v) == 2}

    def _write_info(self):
        if not self.export_dir or not self.items:
            return
        os.makedirs(self.export_dir, exist_ok=True)
        data = {
            "auto_columns": self.chk_auto_cols.isChecked(),
            "rows": self.spn_rows.value(),
            "padding": self.spn_pad.value(),
            "sprite_preview": self.chk_sprite_preview.isChecked(),
            "canvas_auto": self.chk_canvas_auto.isChecked(),
            "canvas_w": self.spn_cw.value(),
            "canvas_h": self.spn_ch.value(),
            "canvas_anchor": list(self._canvas_anchor),
            "canvas_ext_color": self._canvas_ext_color,
            "layer_scales": self._layer_scales,
            "layer_visible": self._layer_visible,
            "layer_offsets": self._layer_offsets,
        }
        try:
            with open(self._info_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
