# -*- coding: utf-8 -*-
"""
模块六：调整位置 - 页面（可嵌入主窗口）

在色调统一之后、导出图片之前，对每一帧做位置与大小的微调：
  - 画布尺寸锁定 = 图片原始大小（不可改），棋盘格表示透明；
  - 视图支持放大镜式缩放（滚轮以鼠标为锚，分辨率不变）、中键平移、
    双击/按钮复位；
  - 每张图片一个独立图层：移动工具拖拽改位置，缩放工具横向拖拽或
    行内数值框改大小（5%~400%）；
  - 框选工具可多选图层（仅可见图层、按非透明包围盒命中、Shift 加选），
    成组后一起移动/缩放；
  - 每张图片独立撤销栈（上限 50 步）：行内 ↩、工具栏按钮或 Ctrl+Z。

调整结果实时烘焙成全分辨率 PNG 输出到 frames_dir/adjusted/（完整序列，
供导出页优先使用），变换参数写入 adjust_info.json 供再次进入时恢复。
输入来源优先级：色调统一 colored/ → 精抠 refined/ → 批量抠图 matted/。
"""

import os
import json
import time
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal, QThread, QRectF, QSize
from PySide6.QtGui import (QKeySequence, QShortcut, QPixmap, QPainter, QColor,
                           QIcon, QPainterPath, QPen)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QGroupBox, QDoubleSpinBox, QToolButton, QFrame, QButtonGroup,
    QFileDialog,
)

from gui_frame_matting import (imread_unicode, imwrite_unicode,
                               frame_no_from_name)
from gui_frame_color import FitImageLabel


# ===================== 图像工具（自包含） =====================

def load_bgra(path: str) -> Optional[np.ndarray]:
    """读图转 BGRA（灰度/3 通道自动补 alpha=255）"""
    img = imread_unicode(path)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def image_size(path: str) -> Optional[Tuple[int, int]]:
    """只读图片宽高（PIL 解析头部），失败回退 cv2"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        pass
    img = imread_unicode(path)
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def scale_bgra(img: np.ndarray, scale: float) -> np.ndarray:
    """等比缩放 BGRA（最小 1px）"""
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if nw == w and nh == h:
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def flatten_over(canvas: np.ndarray, layer: np.ndarray,
                 x: int, y: int) -> np.ndarray:
    """标准 straight-alpha over 合成，支持越界裁剪"""
    ch, cw = canvas.shape[:2]
    lh, lw = layer.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + lw), min(ch, y + lh)
    if x0 >= x1 or y0 >= y1:
        return canvas
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    sub = layer[sy0:sy1, sx0:sx1].astype(np.float32)
    dst = canvas[y0:y1, x0:x1].astype(np.float32)
    sa = sub[:, :, 3:4] / 255.0
    da = dst[:, :, 3:4] / 255.0
    ao = sa + da * (1.0 - sa)
    prem = sub[:, :, :3] * sa + dst[:, :, :3] * da * (1.0 - sa)
    safe = np.clip(ao, 1e-6, None)
    out = np.empty_like(dst)
    out[:, :, :3] = np.where(ao > 1e-6, prem / safe, 0.0)
    out[:, :, 3:4] = ao * 255.0
    canvas[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return canvas


def centered_xy(canvas_w: int, canvas_h: int,
                w: int, h: int) -> Tuple[int, int]:
    return (canvas_w - w) // 2, (canvas_h - h) // 2


def bgra_to_pixmap(bgra: np.ndarray, checker: bool = False) -> QPixmap:
    """BGRA numpy → QPixmap；checker=True 时烘焙棋盘格底"""
    from PySide6.QtGui import QImage
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    qimg = QImage(rgba.data, w, h, 4 * w,
                  QImage.Format_RGBA8888).copy()
    pix = QPixmap.fromImage(qimg)
    if not checker:
        return pix
    from gui_frame_matting import _checker_tile
    canvas = QPixmap(pix.size())
    p = QPainter(canvas)
    p.drawTiledPixmap(canvas.rect(), _checker_tile(12))
    p.drawPixmap(0, 0, pix)
    p.end()
    return canvas


# ===================== 可交互工作台画布 =====================

class WorkspaceCanvas(FitImageLabel):
    """PS 式工作台：拖拽换算成全分辨率画布坐标增量；放大镜式视图缩放"""

    drag_started = Signal()
    drag_delta = Signal(float, float)   # (dx, dy) 全分辨率画布像素
    drag_finished = Signal()
    resized = Signal()
    marquee_changed = Signal(QRectF, bool)  # 画布坐标矩形, additive=Shift
    marquee_done = Signal(QRectF, bool)

    ZOOM_MIN, ZOOM_MAX = 0.05, 20.0     # 5% ~ 2000%

    def __init__(self, checker: bool = True):
        super().__init__(checker)
        self._disp_rect = None           # 图像在控件中的显示矩形
        self.canvas_wh = (1, 1)          # 当前画布全分辨率宽高（换算用）
        self._last = None
        self.tool = "move"               # "move" / "scale" / "marquee"
        # 分层：below / selected / above 三张 pixmap，拖拽时只换 selected；
        # 整组拖拽时改用 _dyn 动态层列表（各自 pixmap 浮于静态链顶层）
        self._layer_below = None
        self._layer_above = None
        self._layer_sel = None
        self._sel_rect = None            # (x,y,w,h) 预览画布坐标
        self._dyn: list = []             # [(pm, (x,y,w,h)), ...] 动态层
        self._rect_key = None
        self._scaled_size = None
        self._checker_tile = None
        # 视图缩放：1.0 = 适配窗口；滚轮锚点缩放，中键平移
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_last = None
        self.zoom_changed = None         # 页面赋值为回调(percent:int)
        self._marquee_start = None
        self._marquee_cur = None
        self.setCursor(Qt.OpenHandCursor)

    def set_tool(self, tool: str):
        self.tool = tool
        self.setCursor(
            Qt.OpenHandCursor if tool == "move"
            else Qt.SizeAllCursor if tool == "scale"
            else Qt.CrossCursor)

    def set_image(self, pix):
        """清空/普通模式：退出分层模式并复位视图缩放"""
        self._layer_below = None
        self._layer_above = None
        self._layer_sel = None
        self._sel_rect = None
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._notify_zoom()
        super().set_image(pix)

    def set_workspace_layers(self, anchor, below, above, sel_pm, sel_rect,
                             dyn=None):
        self._layer_below = below
        self._layer_above = above
        self._layer_sel = sel_pm
        self._sel_rect = sel_rect
        self._dyn = dyn or []
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
        """预览画布在控件中的等比显示矩形（含视图缩放/平移）"""
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
        # 保持鼠标下的画布点不动
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
            p.drawPixmap(rect, self._pix,
                         QRectF(0, 0, self._pix.width(), self._pix.height()))
            p.end()
            return
        pw = self._layer_below.width()
        ph = self._layer_below.height()
        full = QRectF(0, 0, pw, ph)
        p.drawPixmap(rect, self._layer_below, full)
        kx = rect.width() / pw
        ky = rect.height() / ph
        if self._layer_sel is not None and self._sel_rect is not None:
            sx, sy, sw, sh = self._sel_rect
            sel_target = QRectF(rect.x() + sx * kx, rect.y() + sy * ky,
                                sw * kx, sh * ky)
            p.drawPixmap(sel_target, self._layer_sel, QRectF(0, 0, sw, sh))
        for pm, rc in self._dyn:                 # 整组拖拽动态层（浮于顶层）
            dx, dy, dw, dh = rc
            p.drawPixmap(QRectF(rect.x() + dx * kx, rect.y() + dy * ky,
                                dw * kx, dh * ky), pm, QRectF(0, 0, dw, dh))
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
        if self._pix is None or self._pix.isNull():
            self._disp_rect = None
            self.clear()
            self.update()
            return
        self._rect_key = None
        self.update()

    def _screen_to_canvas_rect(self, p0, p1) -> QRectF:
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


# ===================== 预览加载线程 =====================

class _PreviewLoader(QThread):
    """把所有帧解码并缩小缓存，避免预览每次读全分辨率"""
    loaded_one = Signal(str, object, float)   # key, 缓存BGRA, 缓存/原尺寸系数
    done = Signal()

    def __init__(self, items: List[dict], max_side: int = 1000):
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
            img = load_bgra(it["src"])
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


# ===================== 调整结果烘焙线程 =====================

class _BakeWorker(QThread):
    """把图层变换（缩放+偏移）烘焙成全分辨率 PNG 写入 adjusted/"""
    done = Signal(int, int)   # ok, fail

    def __init__(self, tasks: List[tuple], canvas_wh: Tuple[int, int],
                 parent=None):
        # tasks: [(key, src_path, dst_path, scale_pct, ox, oy), ...]
        super().__init__(parent)
        self.tasks = tasks
        self.canvas_wh = canvas_wh
        self._stop = False
        self.idx = 0                 # 已处理到的下标（中断后剩余可重排）

    def stop(self):
        self._stop = True

    def run(self):
        cw, ch = self.canvas_wh
        ok = fail = 0
        for i, (key, src, dst, pct, ox, oy) in enumerate(self.tasks):
            if self._stop:
                break
            self.idx = i
            try:
                img = load_bgra(src)
                if img is None:
                    raise IOError(f"读取失败 {src}")
                sim = scale_bgra(img, pct / 100.0)
                canvas = np.zeros((ch, cw, 4), np.uint8)
                x, y = centered_xy(cw, ch, sim.shape[1], sim.shape[0])
                flatten_over(canvas, sim,
                             x + int(round(ox)), y + int(round(oy)))
                tmp = dst + ".tmp.png"      # imencode 需要 .png 扩展名
                if not imwrite_unicode(tmp, canvas):
                    raise IOError(f"写入失败 {dst}")
                os.replace(tmp, dst)         # 原子替换，防中途退出留残
                ok += 1
            except Exception:
                fail += 1
            self.idx = i + 1
        self.done.emit(ok, fail)


# ===================== 小眼睛矢量图标 =====================

_EYE_ICONS: Dict[bool, "QIcon"] = {}

def eye_icon(visible: bool, px: int = 18) -> "QIcon":
    """矢量小眼睛：显示=亮色睁眼，隐藏=暗色眼+斜杠（替代观感差的 emoji）"""
    icon = _EYE_ICONS.get(visible)
    if icon is not None:
        return icon
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    color = QColor("#ecf0f1") if visible else QColor("#7f8c8d")
    pen = QPen(color, max(1.3, px / 13.0), Qt.SolidLine,
               Qt.RoundCap, Qt.RoundJoin)
    s = px / 24.0                       # 24×24 视口等比映射
    path = QPainterPath()               # 杏仁形眼眶（左右尖角）
    path.moveTo(3 * s, 12 * s)
    path.quadTo(12 * s, 4.5 * s, 21 * s, 12 * s)
    path.quadTo(12 * s, 19.5 * s, 3 * s, 12 * s)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    r = 3.1 * s                         # 瞳孔
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawEllipse(QRectF(12 * s - r, 12 * s - r, 2 * r, 2 * r))
    if not visible:                     # 斜杠
        p.setBrush(Qt.NoBrush)
        p.setPen(pen)
        p.drawLine(int(4.5 * s), int(19.5 * s),
                   int(19.5 * s), int(4.5 * s))
    p.end()
    icon = QIcon(pm)
    _EYE_ICONS[visible] = icon
    return icon


# ===================== 图层列表行 =====================

class LayerRow(QFrame):
    """眼睛显示开关 + 缩略图 + 名称尺寸 + 撤销 + 独立缩放%"""
    clicked = Signal(str)
    scale_changed = Signal(str, float)
    visibility_changed = Signal(str, bool)
    undo_clicked = Signal(str)

    def __init__(self, key: str, number: int, w: int, h: int,
                 pct: int, visible: bool = True):
        super().__init__()
        self.key = key
        self._selected = False
        self._marqueed = False
        self._visible = True
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(6)

        self.eye = QToolButton()
        self.eye.setCheckable(True)
        self.eye.setFixedSize(28, 28)
        self.eye.setCursor(Qt.PointingHandCursor)
        self.eye.setToolTip("显示 / 隐藏该图片（隐藏后不能编辑，"
                            "不参与框选与整组操作）")
        self.eye.setIconSize(QSize(18, 18))
        self.eye.setStyleSheet(
            "QToolButton{background:transparent;border:none;}"
            "QToolButton:hover{background:#3a3a3a;border-radius:4px;}")
        self.eye.toggled.connect(self._on_eye)
        lay.addWidget(self.eye)

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

        self.spin = QDoubleSpinBox()
        self.spin.setRange(5.0, 400.0)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(1.0)
        self.spin.setValue(float(pct))
        self.spin.setSuffix(" %")
        self.spin.setFixedWidth(86)
        self.spin.setCursor(Qt.ArrowCursor)
        self.spin.setToolTip("该图层独立缩放（精确到 0.1%）")
        self.spin.valueChanged.connect(
            lambda v: self.scale_changed.emit(self.key, float(v)))
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
        self.eye.setIcon(eye_icon(visible))
        self.spin.setEnabled(visible)      # 隐藏图层不能编辑
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

    def set_marqueed(self, m: bool):
        """框选多选标记：蓝色高亮（无复选框）"""
        if m != self._marqueed:
            self._marqueed = m
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


# ===================== 参考图层行 =====================

class RefRow(QFrame):
    """参考图层行：只读，仅眼睛控制显隐。
    参考图不参与编辑 / 框选 / 撤销 / 烘焙导出，仅作画布对照"""

    visibility_changed = Signal(bool)

    def __init__(self, visible: bool = True):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedHeight(58)
        self._visible = visible

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)

        self.eye = QToolButton()
        self.eye.setFixedSize(26, 26)
        self.eye.setCursor(Qt.PointingHandCursor)
        self.eye.setToolTip("显示 / 隐藏参考图（参考图不能编辑，不参与导出）")
        self.eye.setIconSize(QSize(16, 16))
        self.eye.setStyleSheet(
            "QToolButton{background:transparent;border:none;}"
            "QToolButton:hover{background:#3a3f4b;border-radius:4px;}")
        self.eye.clicked.connect(self._on_eye)
        lay.addWidget(self.eye)

        self.thumb = QLabel()
        self.thumb.setFixedSize(64, 40)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("background:#111;border-radius:2px;")
        lay.addWidget(self.thumb)

        name = QLabel("参考图")
        name.setStyleSheet(
            "color:#b98fd4;font-weight:bold;background:transparent;")
        lay.addWidget(name, 1)

        self._update_eye()
        self.setStyleSheet(
            "RefRow{background:#2a2530;border:2px solid #554a66;"
            "border-radius:3px;}")

    def is_visible(self) -> bool:
        return self._visible

    def _on_eye(self):
        self.set_visible(not self._visible, emit_signal=True)

    def set_visible(self, visible: bool, emit_signal: bool = False):
        self._visible = visible
        self._update_eye()
        if emit_signal:
            self.visibility_changed.emit(visible)

    def _update_eye(self):
        self.eye.setIcon(eye_icon(self._visible))

    def set_thumbnail(self, bgra: np.ndarray):
        pm = bgra_to_pixmap(bgra)
        if pm is None or pm.isNull():
            return
        self.thumb.setPixmap(pm.scaled(
            self.thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ===================== 调整位置页面 =====================

class FrameAdjustPage(QWidget):

    PREVIEW_MAX = 1500       # 工作台预览最长边像素
    PREVIEW_CACHE = 1000     # 单帧预览缓存最长边（过小会放大模糊）
    UNDO_LIMIT = 50

    def __init__(self, shared_state: dict):
        super().__init__()
        self.shared_state = shared_state
        self.setStyleSheet("QWidget{background:#2b2b2b;color:#ecf0f1;}")

        # 数据
        self.frames_dir = ""
        self.matted_dir = self.refined_dir = self.colored_dir = ""
        self.adjusted_dir = ""
        self.items: List[dict] = []
        self._canvas = (0, 0)                # 锁定画布 (w,h) = 图片原始大小
        self._batch_id = ""                  # 帧集身份（控制变换重置）
        self._reload_sig = ""                # 重载签名（控制是否需重载）
        self.preview_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._alpha_bbox: Dict[str, Tuple[int, int, int, int]] = {}
        self.selected_key: Optional[str] = None
        self.selected_keys: set = set()
        self.layer_rows: List[LayerRow] = []
        self._layer_scales: Dict[str, float] = {}
        self._layer_visible: Dict[str, bool] = {}
        self._layer_offsets: Dict[str, List[float]] = {}   # 相对居中偏移
        self._undo: Dict[str, List[Tuple[float, float, float]]] = {}
        self._undo_coalesce: Dict[str, float] = {}
        self._drag_pending: Optional[str] = None
        self._dragging: bool = False
        self._ws_cache: Optional[dict] = None
        self._dirty: set = set()             # 待重新烘焙的图层
        self._ref_img: Optional[np.ndarray] = None    # 参考图原图 BGRA
        self._ref_canvas: Optional[np.ndarray] = None # 适配画布后的参考图
        self._ref_visible = True
        self._ref_rev = 0                    # 参考图版本号（静态签名用）
        self._ref_session_purged = False     # 本次运行是否已清掉上一会话参考图
        self.ref_row: Optional[RefRow] = None
        self.loader: Optional[_PreviewLoader] = None
        self._bake_worker: Optional[_BakeWorker] = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(24)
        self._preview_timer.timeout.connect(self._render_workspace)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._flush_dirty)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._on_view_resized)
        # 键盘连调（WASD/QE）期间把目标图层当动态层渲染，停手后恢复静态
        self._key_idle_timer = QTimer(self)
        self._key_idle_timer.setSingleShot(True)
        self._key_idle_timer.setInterval(250)
        self._key_idle_timer.timeout.connect(self._end_key_drag)

        self._build_ui()
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo_selected)
        # WASD 微调位置（1px/次），QE 调大小（步进用户可设，默认1%）
        QShortcut(QKeySequence(Qt.Key_W), self, context=Qt.WindowShortcut,
                  activated=lambda: self._key_nudge(0, -1))
        QShortcut(QKeySequence(Qt.Key_A), self, context=Qt.WindowShortcut,
                  activated=lambda: self._key_nudge(-1, 0))
        QShortcut(QKeySequence(Qt.Key_S), self, context=Qt.WindowShortcut,
                  activated=lambda: self._key_nudge(0, 1))
        QShortcut(QKeySequence(Qt.Key_D), self, context=Qt.WindowShortcut,
                  activated=lambda: self._key_nudge(1, 0))
        QShortcut(QKeySequence(Qt.Key_Q), self, context=Qt.WindowShortcut,
                  activated=lambda: self._key_scale(-1))
        QShortcut(QKeySequence(Qt.Key_E), self, context=Qt.WindowShortcut,
                  activated=lambda: self._key_scale(1))

    # ---------- UI ----------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ===== 左侧：工作台 =====
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

        self.info_label = QLabel("尚未加载图片")
        self.info_label.setStyleSheet("color:#f1c40f;font-size:12px;")
        lv.addWidget(self.info_label)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;font-size:12px;")
        lv.addWidget(self.status)
        root.addWidget(left_box, 13)

        # ===== 右侧：工具 + 图层列表 =====
        right = QVBoxLayout()
        right.setSpacing(8)

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
        btn_zero.setToolTip("选中图片回到画布中心（可撤销，框选组整组归零）")
        btn_zero.clicked.connect(self._reset_selected_offset)
        rowt.addWidget(btn_zero)
        self.btn_undo_sel = QPushButton("↩ 撤销本图")
        self.btn_undo_sel.setToolTip(
            "撤销选中图片的上一次移动/缩放（Ctrl+Z，各图独立）")
        self.btn_undo_sel.clicked.connect(self._undo_selected)
        self.btn_undo_sel.setEnabled(False)
        rowt.addWidget(self.btn_undo_sel)
        rowt.addWidget(QLabel(" QE步进:"))
        self.key_step_spin = QDoubleSpinBox()
        self.key_step_spin.setRange(0.1, 10.0)
        self.key_step_spin.setSingleStep(0.1)
        self.key_step_spin.setDecimals(1)
        self.key_step_spin.setValue(1.0)
        self.key_step_spin.setSuffix("%")
        self.key_step_spin.setToolTip(
            "Q/E 键每按一次的缩放变化量（0.1%~10%，默认1%）\n"
            "W/A/S/D 键每按一次移动 1px（均作用于当前操作图/框选组）")
        rowt.addWidget(self.key_step_spin)
        rowt.addStretch()
        tf.addLayout(rowt)

        # 视图缩放（放大镜式，分辨率不变）
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

        rowc = QHBoxLayout()
        self.canvas_label = QLabel("画布尺寸：-")
        self.canvas_label.setStyleSheet("color:#2ecc71;")
        self.canvas_label.setToolTip("画布锁定为图片原始大小，不可修改；"
                                     "视图缩放只是放大镜，不改变分辨率")
        rowc.addWidget(self.canvas_label)
        rowc.addStretch()
        tf.addLayout(rowc)
        right.addWidget(tool_box, 0)

        # ---- 图层列表 ----
        list_box = QGroupBox("所有图片（眼睛=是否显示，点击行选中）")
        gl = QVBoxLayout(list_box)
        gl.setContentsMargins(4, 6, 4, 4)
        head = QHBoxLayout()
        self.eye_all = QToolButton()
        self.eye_all.setIcon(eye_icon(True))
        self.eye_all.setFixedSize(28, 28)
        self.eye_all.setCursor(Qt.PointingHandCursor)
        self.eye_all.setToolTip("全部显示 / 全部隐藏")
        self.eye_all.setIconSize(QSize(18, 18))
        self.eye_all.setStyleSheet(
            "QToolButton{background:transparent;border:none;}"
            "QToolButton:hover{background:#3a3f4b;border-radius:4px;}")
        self.eye_all.clicked.connect(self._toggle_all_visible)
        head.addWidget(self.eye_all)
        head.addWidget(QLabel("图层列表"))
        head.addStretch()
        btn_ref = QPushButton("📌 参考图…")
        btn_ref.setToolTip("上传一张参考图作为对照（不能编辑，不参与导出）")
        btn_ref.clicked.connect(self._upload_reference)
        head.addWidget(btn_ref)
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

    # ---------- 批次/加载 ----------

    def showEvent(self, event):
        super().showEvent(event)
        self._reload_if_changed()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._flush_dirty()

    @staticmethod
    def _sig(directory: str) -> str:
        if not directory or not os.path.isdir(directory):
            return ""
        try:
            return "files:" + "|".join(sorted(
                f for f in os.listdir(directory)
                if f.lower().endswith(".png")))
        except OSError:
            return ""

    @staticmethod
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

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
            self._show_empty("尚未找到制作好的图片\n\n"
                             "请先完成批量抠图 / 精抠 / 色调统一")
            return
        files = sorted(f for f in os.listdir(matted_dir)
                       if f.lower().endswith(".png"))
        if not files:
            self._show_empty("批量抠图目录中没有 PNG 图片")
            return

        # 帧集身份：换视频/重新抽帧/重新批量抠图 → 重置全部变换
        batch_id = "|".join((
            os.path.abspath(frames_dir) if frames_dir else "",
            os.path.abspath(matted_dir),
            str(self._mtime(os.path.join(frames_dir, "frames_info.json"))
                if frames_dir else ""),
            str(self._mtime(os.path.join(matted_dir, "matted_info.json")))))
        # 重载签名：帧集 + 上游内容变化（重精抠/重色调 → 保留变换，重烘焙）
        reload_sig = "|".join((
            batch_id, self._sig(colored_dir), self._sig(refined_dir),
            str(self._mtime(os.path.join(colored_dir, "colored_info.json"))
                if colored_dir else ""),
            str(self._mtime(os.path.join(refined_dir, "refined_info.json"))
                if refined_dir else "")))
        if reload_sig == self._reload_sig:
            return
        batch_changed = bool(self._batch_id) and batch_id != self._batch_id
        self._batch_id = batch_id
        self._reload_sig = reload_sig

        if self.loader is not None and self.loader.isRunning():
            self.loader.stop()
            self.loader.wait(2000)
        if self._bake_worker is not None and self._bake_worker.isRunning():
            self._bake_worker.stop()
            self._bake_worker.wait(2000)

        self.frames_dir, self.matted_dir = frames_dir, matted_dir
        self.refined_dir, self.colored_dir = refined_dir, colored_dir
        self.adjusted_dir = (os.path.join(frames_dir, "adjusted")
                             if frames_dir else
                             os.path.join(os.path.dirname(matted_dir),
                                          "adjusted"))
        os.makedirs(self.adjusted_dir, exist_ok=True)

        self.items = []
        self._undo = {}
        self._undo_coalesce = {}
        self._drag_pending = None
        self._drag_group = None            # 整组拖拽中的动态图层集合
        self._pm_cache = {}                # (key,pct,pf)→(pm,w,h) 动态层缩放缓存
        self._ws_cache = None
        self.selected_keys = set()
        if batch_changed:                    # 帧集变了：变换全部重置
            self._layer_scales = {}
            self._layer_offsets = {}
            self._layer_visible = {}
            self._dirty = set()
        for i, fn in enumerate(files, 1):
            src, stage = self._best_source(fn, colored_dir,
                                           refined_dir, matted_dir)
            size = image_size(src) or (0, 0)
            self.items.append({"key": fn,
                               "number": frame_no_from_name(fn, i),
                               "src": src,
                               "stage": stage, "w": size[0], "h": size[1]})
        # 画布锁定 = 图片原始大小（取最大帧）
        self._canvas = (max(it["w"] for it in self.items),
                        max(it["h"] for it in self.items))
        self.canvas_label.setText(
            f"画布尺寸：{self._canvas[0]}×{self._canvas[1]}（锁定）")

        self._load_reference_file()          # 参考图（仅本次运行内保留）
        if not batch_changed:
            self._restore_info()             # 同批次恢复上次变换

        if self.selected_key not in {it["key"] for it in self.items}:
            self.selected_key = self.items[0]["key"]
        self._rebuild_layer_list()
        self._start_preview_loader()

        # 清理 adjusted/ 中不属于当前帧集的残留 PNG（参考图除外）
        keys = {it["key"] for it in self.items}
        try:
            for f in os.listdir(self.adjusted_dir):
                if f == self.REF_FILE:
                    continue
                if f.lower().endswith(".png") and f not in keys:
                    try:
                        os.remove(os.path.join(self.adjusted_dir, f))
                    except OSError:
                        pass
        except OSError:
            pass
        # 源文件比烘焙产物新（或缺失）→ 重新烘焙该帧
        self._dirty = {it["key"] for it in self.items
                       if self._bake_stale(it)}
        if batch_changed:
            self._dirty = set(keys)
        if self._dirty:
            self._save_timer.start(50)
        self.shared_state["adjusted_dir"] = self.adjusted_dir
        self.status_text(
            f"已加载 {len(self.items)} 张（来源：{self._stage_name()}）"
            + ("，帧集已变化，变换已重置" if batch_changed else ""))

    def _best_source(self, fn: str, colored_dir: str, refined_dir: str,
                     matted_dir: str) -> Tuple[str, str]:
        """来源优先级 colored → refined → matted"""
        if colored_dir:
            p = os.path.join(colored_dir, fn)
            if os.path.isfile(p):
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

    def _show_empty(self, text: str):
        self.items = []
        self._batch_id = ""
        self._reload_sig = ""
        self.preview_cache = {}
        self._ws_cache = None
        self._drag_group = None
        self._pm_cache = {}
        self._ref_img = None
        self._ref_canvas = None
        self._ref_visible = True
        self._dirty = set()
        self._canvas = (0, 0)
        self.adjusted_dir = ""
        self.canvas_label.setText("画布尺寸：-")
        self._rebuild_layer_list()
        self.view.set_image(None)
        self.info_label.setText(text)

    def _start_preview_loader(self):
        if self.loader is not None and self.loader.isRunning():
            self.loader.stop()
            self.loader.wait(2000)
        self.preview_cache = {}
        self._ws_cache = None
        self._drag_group = None
        self._pm_cache = {}
        self.loader = _PreviewLoader(self.items, self.PREVIEW_CACHE)
        self.loader.loaded_one.connect(self._on_preview_one)
        self.loader.done.connect(self._render_workspace)
        self.loader.start()

    def _on_preview_one(self, key: str, img: np.ndarray, factor: float):
        self.preview_cache[key] = (img, factor)
        # 非透明像素包围盒（换算回全分辨率），供框选命中
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
            self._render_workspace()
        self.status_text(f"正在生成预览 {len(self.preview_cache)}"
                         f"/{len(self.items)}")

    # ---------- 图层列表 ----------

    def _rebuild_layer_list(self):
        while self.layer_layout.count() > 1:      # 保留末尾 stretch
            item = self.layer_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.layer_rows = []
        self.ref_row = None
        if self._ref_img is not None:        # 参考图层固定在列表最上面
            rr = RefRow(self._ref_visible)
            rr.visibility_changed.connect(self._on_ref_visibility)
            rr.set_thumbnail(self._ref_img)
            self.layer_layout.insertWidget(self.layer_layout.count() - 1, rr)
            self.ref_row = rr
        for it in self.items:
            pct = self._layer_scales.get(it["key"], 100.0)
            row = LayerRow(it["key"], it["number"], it["w"], it["h"],
                           int(round(pct)),
                           self._layer_visible.get(it["key"], True))
            row.clicked.connect(self._select_layer)
            row.scale_changed.connect(self._on_layer_scale_changed)
            row.visibility_changed.connect(self._on_layer_visibility)
            row.undo_clicked.connect(self._undo_layer)
            self.layer_layout.insertWidget(self.layer_layout.count() - 1,
                                           row)
            self.layer_rows.append(row)
            if it["key"] in self.preview_cache:
                row.set_thumbnail(self.preview_cache[it["key"]][0])
        self._sync_row_selection()
        self._update_undo_ui()

    def _select_layer(self, key: str):
        self.selected_key = key
        self._clear_marquee()
        self._sync_row_selection()
        self._update_undo_ui()
        self._render_workspace()

    def _sync_row_selection(self):
        for r in self.layer_rows:
            r.set_selected(r.key == self.selected_key)

    # ---------- 参考图层 ----------

    REF_FILE = "_reference.png"

    def _build_ref_canvas(self):
        """把参考图按比例适配到锁定画布（居中，不同分辨率等比缩放）"""
        self._ref_canvas = None
        if self._ref_img is None:
            return
        cw, ch = self._canvas
        if cw <= 0 or ch <= 0:
            return
        img = self._ref_img
        h, w = img.shape[:2]
        s = min(cw / w, ch / h)
        if abs(s - 1.0) > 1e-6:
            img = scale_bgra(img, s)
        canvas = np.zeros((ch, cw, 4), np.uint8)
        x, y = centered_xy(cw, ch, img.shape[1], img.shape[0])
        flatten_over(canvas, img, x, y)
        self._ref_canvas = canvas
        self._ref_rev += 1

    def _load_reference_file(self):
        """恢复本次运行内上传的参考图。

        参考图不跨程序运行保留：启动后首次进入页面时，先删除上一会话
        残留的 adjusted/_reference.png；本次运行内上传的副本仍可跨批次恢复。"""
        self._ref_img = None
        self._ref_canvas = None
        if not self.adjusted_dir:
            return
        path = os.path.join(self.adjusted_dir, self.REF_FILE)
        if not self._ref_session_purged:
            self._ref_session_purged = True
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return
        if os.path.isfile(path):
            img = load_bgra(path)
            if img is not None:
                self._ref_img = img
                self._build_ref_canvas()

    def _upload_reference(self):
        if not self.items:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        img = load_bgra(path)
        if img is None:
            self.status_text("⚠ 参考图读取失败")
            return
        self._ref_img = img
        self._build_ref_canvas()
        self._ref_visible = True
        if self.adjusted_dir:
            imwrite_unicode(os.path.join(self.adjusted_dir, self.REF_FILE),
                            img)
        self._ws_cache = None
        self._rebuild_layer_list()
        self._render_workspace()
        self._write_info()
        h, w = img.shape[:2]
        cw, ch = self._canvas
        fit = "" if (w, h) == (cw, ch) else f"，已按比例适配画布 {cw}×{ch}"
        self.status_text(f"📌 参考图已加载 {w}×{h}{fit}")

    def _on_ref_visibility(self, visible: bool):
        self._ref_visible = visible
        self._ws_cache = None
        self._render_workspace()
        self._write_info()

    # ---------- 变换与撤销 ----------

    def _layer_transform(self, key: str) -> Tuple[float, float, float]:
        pct = float(self._layer_scales.get(key, 100.0))
        ox, oy = self._layer_offsets.get(key, (0.0, 0.0))
        return pct, float(ox), float(oy)

    def _push_undo(self, key: str, mark_coalesce: bool = True):
        """把该图【当前】变换状态作为撤销点入栈（在改动之前调用）"""
        snap = self._layer_transform(key)
        st = self._undo.setdefault(key, [])
        if st and st[-1] == snap:
            return
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
            self._undo.pop(key, None)
        self._layer_scales[key] = pct
        if ox == 0.0 and oy == 0.0:
            self._layer_offsets.pop(key, None)
        else:
            self._layer_offsets[key] = [ox, oy]
        row = next((r for r in self.layer_rows if r.key == key), None)
        if row is not None:
            disp = max(row.spin.minimum(),
                       min(row.spin.maximum(), float(pct)))
            row.spin.blockSignals(True)
            row.spin.setValue(disp)
            row.spin.blockSignals(False)
        # 撤销是一个明确的断点：之后的任何改动都开新撤销点
        # （若沿用合并窗口，撤销后 0.8s 内的改动会丢失"回到撤销后状态"的快照）
        self._undo_coalesce[key] = 0.0
        self._ws_cache = None
        self._dirty.add(key)
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

    def _on_layer_scale_changed(self, key: str, pct: float):
        # 键盘连按/滚轮/拖拽产生的连续变化合并为一个撤销点
        now = time.monotonic()
        if now - self._undo_coalesce.get(key, 0.0) > 0.8:
            self._push_undo(key)
        self._undo_coalesce[key] = now
        self._layer_scales[key] = float(pct)
        self._dirty.add(key)
        if key == self.selected_key:
            self._render_workspace()
        else:
            self._schedule_preview()
        self._save_timer.start()

    def _on_layer_visibility(self, key: str, visible: bool):
        self._layer_visible[key] = visible
        row = next((r for r in self.layer_rows if r.key == key), None)
        if row is not None and row.is_visible() != visible:
            row.set_visible(visible)           # 同步行状态（禁用缩放等）
        if not visible and key in self.selected_keys:
            # 隐藏后退出多选组：不随整组变换，蓝色高亮同步消失
            self.selected_keys.discard(key)
            if row is not None:
                row.set_marqueed(False)
        self._schedule_preview()
        self._save_timer.start()

    def _set_all_visible(self, visible: bool):
        for r in self.layer_rows:
            r.set_visible(visible)
            self._layer_visible[r.key] = visible
        if not visible:
            self.selected_keys = set()       # 全隐藏：多选组清空
            for r in self.layer_rows:
                r.set_marqueed(False)
        self.eye_all.setIcon(eye_icon(visible))
        self._schedule_preview()
        self._save_timer.start()
        self.status_text("全部图层已显示" if visible else "全部图层已隐藏")

    def _toggle_all_visible(self):
        any_visible = any(
            self._layer_visible.get(r.key, True) for r in self.layer_rows)
        self._set_all_visible(not any_visible)

    # ---------- 框选多选 ----------

    def _workspace_preview_factor(self, cw: int, ch: int) -> float:
        """预览分辨率按控件实际显示尺寸走（封顶 PREVIEW_MAX）"""
        vw = max(1, self.view.width())
        vh = max(1, self.view.height())
        target = min(float(self.PREVIEW_MAX), float(max(vw, vh)))
        return min(1.0, target / max(1.0, float(max(cw, ch))))

    def _marquee_hit(self, rect, it) -> bool:
        """图层非透明内容在画布上的实际位置是否与框选矩形相交"""
        key = it["key"]
        bbox = self._alpha_bbox.get(key)
        if bbox is None:
            bbox = (0, 0, it["w"], it["h"])
        l, t, r, b = bbox
        ox, oy = self._layer_offsets.get(key, (0.0, 0.0))
        cw, ch = self._canvas
        pf = self._workspace_preview_factor(cw, ch)
        pw, ph = int(cw * pf), int(ch * pf)
        k = self._layer_scales.get(key, 100.0) / 100.0
        fw, fh = it["w"] * k * pf, it["h"] * k * pf
        fx, fy = centered_xy(pw, ph, fw, fh)
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
        if self.selected_key:
            self._drag_pending = self.selected_key
        self._dragging = True
        # 拖拽期间不烘焙：停掉待发的保存定时器，中断后台烘焙队列
        # （否则松手后的全分辨率烘焙与下一次拖拽抢 CPU，起步卡顿）
        self._save_timer.stop()
        w = self._bake_worker
        if w is not None and w.isRunning():
            w.stop()
            for t in w.tasks[w.idx:]:        # 未完成的重新排队
                self._dirty.add(t[0])

    def _on_drag_finished(self):
        self._drag_pending = None
        self._dragging = False
        if self._drag_group is not None:
            self._drag_group = None
            self._render_workspace()       # 落定后按真实层序完整重绘
        self._save_timer.start()

    def _on_view_resized(self):
        if self.items:
            self._ws_cache = None      # 预览分辨率随控件变化，缓存失效
            self._pm_cache = {}
            self._render_workspace()

    def _drag_keys(self) -> List[str]:
        """本次拖拽作用的目标图层：主选优先；若在框选组内则整组（仅可见）。
        当前图层被隐藏时不可编辑，返回空"""
        key = self.selected_key
        if not key or not self._layer_visible.get(key, True):
            return []
        if self.selected_keys and key in self.selected_keys:
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
            self._drag_group = frozenset(keys)   # 整组作为动态层
            for k in keys:
                ox, oy = self._layer_offsets.get(k, (0.0, 0.0))
                self._layer_offsets[k] = [ox + dx, oy + dy]
                self._dirty.add(k)
            # 拖拽中静态层+动态缩放均命中缓存，立即跟手渲染
            self._render_workspace()
        else:
            self._drag_group = frozenset(keys)
            for k in keys:
                row = next((r for r in self.layer_rows
                            if r.key == k), None)
                if row is not None:
                    cur = row.spin.value()
                    new = max(5.0, min(400.0, round(cur + dx * 0.5, 1)))
                    if new != cur:
                        row.spin.setValue(new)  # 触发 changed → 预览+保存

    def _reset_selected_offset(self):
        keys = self._drag_keys()
        if not keys:
            return
        changed = 0
        for k in keys:
            ox, oy = self._layer_offsets.get(k, (0.0, 0.0))
            if ox == 0.0 and oy == 0.0:
                continue
            self._push_undo(k)
            self._layer_offsets[k] = [0.0, 0.0]
            self._dirty.add(k)
            changed += 1
        if changed:
            self._ws_cache = None
            self._render_workspace()
            self._save_timer.start()
            self.status_text(f"{changed} 张图片已回到画布中心")

    # ---------- 键盘微调（WASD 移动 / QE 缩放） ----------

    def _key_targets(self) -> List[str]:
        """键盘控制的目标图层（同拖拽语义：当前图/框选组，隐藏不可编辑）"""
        keys = self._drag_keys()
        if not keys and self.selected_key:
            self.status_text("（已隐藏，隐藏的图层不能编辑）")
        return keys

    def _begin_key_drag(self, keys: List[str]):
        """键盘连调期间目标图层按动态层渲染（命中缓存，不重建静态链）"""
        self._drag_group = frozenset(keys)
        self._key_idle_timer.start()      # 停手 250ms 后 _end_key_drag 恢复

    def _end_key_drag(self):
        if self._drag_group:
            self._drag_group = frozenset()
            self._render_workspace()

    def _key_nudge(self, dx: float, dy: float):
        """WASD：当前图（或框选组）每次移动 1px，连按合并一个撤销点"""
        keys = self._key_targets()
        if not keys:
            return
        now = time.monotonic()
        for k in keys:
            if now - self._undo_coalesce.get(k, 0.0) > 0.8:
                self._push_undo(k)
            self._undo_coalesce[k] = now
            ox, oy = self._layer_offsets.get(k, (0.0, 0.0))
            self._layer_offsets[k] = [ox + dx, oy + dy]
            self._dirty.add(k)
        self._begin_key_drag(keys)
        self._render_workspace()
        self._save_timer.start()

    def _key_scale(self, direction: int):
        """QE：当前图（或框选组）缩放 ±步进%（浮点累加，5%~400% 封顶）"""
        keys = self._key_targets()
        if not keys:
            return
        step = self.key_step_spin.value() * direction
        now = time.monotonic()
        changed = False
        last_pct = None
        for k in keys:
            old = self._layer_scales.get(k, 100.0)
            new = max(5.0, min(400.0, old + step))
            if new == old:
                continue
            if now - self._undo_coalesce.get(k, 0.0) > 0.8:
                self._push_undo(k)
            self._undo_coalesce[k] = now
            self._layer_scales[k] = new
            self._dirty.add(k)
            row = next((r for r in self.layer_rows if r.key == k), None)
            if row is not None:      # spin 与浮点值同步（0.1% 精度）
                row.spin.blockSignals(True)
                row.spin.setValue(max(row.spin.minimum(),
                                      min(row.spin.maximum(),
                                          float(new))))
                row.spin.blockSignals(False)
            last_pct = new
            changed = True
        if changed:
            self._begin_key_drag(keys)
            self._render_workspace()
            self._save_timer.start()
            self.status_text(f"缩放 {last_pct:.1f}%")

    # ---------- 工作台渲染 ----------

    def _schedule_preview(self, *a):
        self._preview_timer.start()

    def _place(self, it: dict, cache: dict, pw: int, ph: int, pf: float):
        """缩放后图层 + 它在预览画布上的左上角坐标"""
        im, cf = cache[it["key"]]
        pct = self._layer_scales.get(it["key"], 100.0) / 100.0
        sim = scale_bgra(im, pct * pf / cf)
        x, y = centered_xy(pw, ph, sim.shape[1], sim.shape[0])
        ox, oy = self._layer_offsets.get(it["key"], (0.0, 0.0))
        return sim, x + int(round(ox * pf)), y + int(round(oy * pf))

    def _dyn_pixmap(self, it: dict, cache: dict, pf: float):
        """拖拽动态层的缩放结果缓存：移动拖拽时复用 pixmap 只平移，
        仅缩放变化时才重采样该层"""
        k = it["key"]
        im, cf = cache[k]
        pct = self._layer_scales.get(k, 100.0)
        qkey = (k, round(pct, 3), round(pf, 4))
        hit = self._pm_cache.get(qkey)
        if hit is None:
            sim = scale_bgra(im, pct / 100.0 * pf / cf)
            hit = (bgra_to_pixmap(sim, checker=False),
                   sim.shape[1], sim.shape[0])
            for old in [q for q in self._pm_cache if q[0] == k]:
                del self._pm_cache[old]      # 同 key 旧规格作废
            self._pm_cache[qkey] = hit
        return hit

    def _render_workspace(self):
        """合成所有可见图层（含独立缩放与位置偏移），静态层缓存"""
        if not self.items:
            self.view.set_image(None)
            return
        cache = self.preview_cache
        if not cache:
            return
        cw, ch = self._canvas
        self.view.canvas_wh = (cw, ch)
        pf = self._workspace_preview_factor(cw, ch)
        pw = max(1, int(cw * pf))
        ph = max(1, int(ch * pf))
        sel = self.selected_key
        sel_idx = next(
            (i for i, it in enumerate(self.items) if it["key"] == sel), -1)

        def sig_of(it):
            k = it["key"]
            off = self._layer_offsets.get(k, (0.0, 0.0))
            return (k, self._layer_scales.get(k, 100.0),
                    round(float(off[0]), 2), round(float(off[1]), 2),
                    self._layer_visible.get(k, True))

        # 整组拖拽中：拖拽层全部作为动态层，静态链只含未拖拽层。
        # 签名用「被排除的图层集合」而不是拖拽状态——单图层拖拽时被排除的
        # 就是选中层本身，与拖拽前完全一致 → 静态缓存直接命中，
        # 拖拽开始/结束都不再触发全链重建（修复拖动首帧卡顿）。
        dyn_keys = self._drag_group or frozenset()
        static_items = [it for it in self.items
                        if it["key"] != sel and it["key"] not in dyn_keys]
        excluded = set(dyn_keys)
        if sel:
            excluded.add(sel)
        static_sig = (
            pw, ph, tuple(sorted(excluded)),
            tuple(sorted(cache.keys())),
            ("ref", self._ref_rev, self._ref_visible),
            tuple(sig_of(it) for it in static_items))
        wc = self._ws_cache
        if wc is None or wc.get("sig") != static_sig:
            shown = 0
            # 单图层拖拽（excluded 只有选中层）仍按真实层序分上下链；
            # 整组拖拽按组的层序区间切分：低于组→below，高于组→above，
            # 组内夹层静态层渲染时与动态层按层序交错（不再整体置顶，
            # 修复拖动底层图时它短暂出现在顶层的问题）
            group = bool(set(dyn_keys) - {sel})
            g_lo = g_hi = -1
            if group:
                gi = [i for i, it in enumerate(self.items)
                      if it["key"] in dyn_keys]
                g_lo, g_hi = min(gi), max(gi)
            middle_keys: List[str] = []
            # QPainter SourceOver（C++ 光栅）合成，比 numpy 快一个量级；
            # 每层 pixmap 复用 _pm_cache（pct/pf 不变时跨重建命中）
            below_pm = QPixmap(pw, ph)
            below_pm.fill(Qt.transparent)
            above_pm = None
            p_below = QPainter(below_pm)
            p_above = None
            if self._ref_canvas is not None and self._ref_visible:
                # 参考图层垫在最底层（仅对照，不参与烘焙导出）
                p_below.drawPixmap(0, 0, bgra_to_pixmap(
                    scale_bgra(self._ref_canvas, pf), checker=False))
            for i, it in enumerate(self.items):
                k = it["key"]
                if k == sel or k in dyn_keys:
                    continue
                if cache.get(k) is None:
                    continue
                if not self._layer_visible.get(k, True):
                    continue
                pm, w, h = self._dyn_pixmap(it, cache, pf)
                x, y = centered_xy(pw, ph, w, h)
                ox, oy = self._layer_offsets.get(k, (0.0, 0.0))
                px = x + int(round(ox * pf))
                py = y + int(round(oy * pf))
                if group:
                    if i < g_lo:
                        p_below.drawPixmap(px, py, pm)
                        shown += 1
                    elif i > g_hi:
                        if above_pm is None:
                            above_pm = QPixmap(pw, ph)
                            above_pm.fill(Qt.transparent)
                            p_above = QPainter(above_pm)
                        p_above.drawPixmap(px, py, pm)
                        shown += 1
                    else:
                        middle_keys.append(k)   # 组内夹层，渲染时按层序交错
                elif sel_idx < 0 or i < sel_idx:
                    p_below.drawPixmap(px, py, pm)
                    shown += 1
                else:
                    if above_pm is None:
                        above_pm = QPixmap(pw, ph)
                        above_pm.fill(Qt.transparent)
                        p_above = QPainter(above_pm)
                    p_above.drawPixmap(px, py, pm)
                    shown += 1
            p_below.end()
            if p_above is not None:
                p_above.end()
            anchor = QPixmap(pw, ph)
            anchor.fill(Qt.transparent)
            wc = {"sig": static_sig,
                  "anchor": anchor,
                  "below_pm": below_pm,
                  "above_pm": above_pm,
                  "middle_keys": middle_keys,
                  "static_shown": shown}
            self._ws_cache = wc

        shown = wc["static_shown"]
        sel_pm = None
        sel_rect = None
        if (not dyn_keys and sel_idx >= 0 and sel in cache
                and self._layer_visible.get(sel, True)):
            sim, px, py = self._place(self.items[sel_idx], cache, pw, ph, pf)
            sel_pm = bgra_to_pixmap(sim, checker=False)
            sel_rect = (px, py, sim.shape[1], sim.shape[0])
            shown += 1
        dyn = []
        if dyn_keys:
            middle = set(wc.get("middle_keys", ()))
            for it in self.items:          # 按层序：拖拽层与组内夹层静态交错
                k = it["key"]
                if k not in dyn_keys and k not in middle:
                    continue
                if k not in cache:
                    continue
                if not self._layer_visible.get(k, True):
                    continue
                pm, w, h = self._dyn_pixmap(it, cache, pf)
                x, y = centered_xy(pw, ph, w, h)
                ox, oy = self._layer_offsets.get(k, (0.0, 0.0))
                dyn.append((pm, (x + int(round(ox * pf)),
                                 y + int(round(oy * pf)), w, h)))
                shown += 1
        self.view.set_workspace_layers(
            wc["anchor"], wc["below_pm"], wc["above_pm"],
            sel_pm, sel_rect, dyn)
        total = len(self.items)
        hidden = total - sum(1 for it in self.items
                             if self._layer_visible.get(it["key"], True))
        sel_hidden = (self.selected_key
                      and not self._layer_visible.get(self.selected_key, True))
        self.info_label.setText(
            f"工作台 {cw}×{ch}：{shown} 个可见图层"
            + (f"，{hidden} 个已隐藏" if hidden else "")
            + f"｜选中：{self.selected_key or '-'}"
            + ("（已隐藏，隐藏的图层不能编辑）" if sel_hidden else ""))

    # ---------- 持久化 ----------

    def _info_path(self) -> str:
        return os.path.join(self.adjusted_dir, "adjust_info.json")

    def _restore_info(self):
        """同批次重新进入：恢复上次的变换（缩放/偏移/可见性）"""
        try:
            with open(self._info_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if data.get("batch") != self._batch_id:
            return
        keys = {it["key"] for it in self.items}
        self._layer_scales = {
            k: max(5.0, min(400.0, float(v)))
            for k, v in data.get("scales", {}).items() if k in keys}
        self._layer_offsets = {
            k: [float(v[0]), float(v[1])]
            for k, v in data.get("offsets", {}).items()
            if k in keys and isinstance(v, (list, tuple)) and len(v) == 2}
        self._layer_visible = {
            k: bool(v) for k, v in data.get("visible", {}).items()
            if k in keys}
        self._ref_visible = bool(data.get("ref_visible", True))

    def _write_info(self):
        if not self.adjusted_dir or not self.items:
            return
        os.makedirs(self.adjusted_dir, exist_ok=True)
        data = {
            "batch": self._batch_id,
            "canvas": list(self._canvas),
            "scales": self._layer_scales,
            "offsets": self._layer_offsets,
            "visible": self._layer_visible,
            "ref_visible": self._ref_visible,
        }
        try:
            with open(self._info_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # ---------- 烘焙（输出 adjusted/ 全分辨率序列） ----------

    def _bake_stale(self, it: dict) -> bool:
        """烘焙产物缺失或比源文件旧 → 需要重新烘焙"""
        dst = os.path.join(self.adjusted_dir, it["key"])
        if not os.path.isfile(dst):
            return True
        src_mt = self._mtime(it["src"])
        return src_mt > 0 and self._mtime(dst) < src_mt - 1e-6

    def _flush_dirty(self):
        if not self.items or not self.adjusted_dir:
            self._dirty = set()
            return
        if self._dragging:                   # 拖拽中：松手后再保存
            self._save_timer.start()
            return
        self._write_info()
        if self._bake_worker is not None and self._bake_worker.isRunning():
            return                       # 运行中；结束后会再次 flush
        if not self._dirty:
            return
        tasks = []
        by_key = {it["key"]: it for it in self.items}
        for key in sorted(self._dirty):
            it = by_key.get(key)
            if it is None:
                continue
            pct, ox, oy = self._layer_transform(key)
            tasks.append((key, it["src"],
                          os.path.join(self.adjusted_dir, key),
                          pct, ox, oy))
        self._dirty = set()
        if not tasks:
            return
        self._bake_worker = _BakeWorker(tasks, self._canvas, self)
        self._bake_worker.done.connect(self._on_bake_done)
        self._bake_worker.start(QThread.LowPriority)   # 让 UI 线程优先
        self.status_text(f"正在保存调整结果 {len(tasks)} 张…")

    def _on_bake_done(self, ok: int, fail: int):
        self.shared_state["adjusted_dir"] = self.adjusted_dir
        self.status_text(f"✅ 调整结果已保存 {ok} 张到 adjusted/"
                         + (f"，失败 {fail} 张" if fail else ""))
        if self._dirty:                  # 烘焙期间又有改动
            self._save_timer.start()

    def status_text(self, text: str):
        self.status.setText(text)
