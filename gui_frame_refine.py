# -*- coding: utf-8 -*-
"""
模块三-补：图片精抠 - 页面（可嵌入主窗口）

对「批量抠图」的结果逐张做人工修正：
  - 橡皮擦：硬性擦除（alpha→0），删除残留杂边
  - 背景橡皮擦：连续取样背景色，只擦除与取样色相近的像素（保护主体）
  - 还原橡皮擦：从未抠图原始帧恢复像素（补回被误删的主体）

每张图片拥有独立的撤销栈，切换图片互不影响；结果输出到 refined/ 目录。

布局与其它页面一致：左侧工作区（可涂抹画布），
右侧上半功能按钮区、下半精抠图片缩略图网格，左右固定 13:8。
"""

import os
import json
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal, QPointF, QRect, QRectF
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSlider, QScrollArea, QMessageBox, QGroupBox,
    QSizePolicy, QRadioButton, QButtonGroup, QFrame,
)

from gui_frame_matting import (
    imread_unicode, imwrite_unicode, _checker_tile,
)


# ===================== 单张图片的编辑状态 =====================

class EditState:
    """一张图片的完整编辑上下文（撤销栈按图片独立保存）"""

    UNDO_LIMIT = 60

    def __init__(self, key: str, filename: str,
                 matted_path: str, orig_path: Optional[str],
                 refined_path: str, base: np.ndarray,
                 orig: Optional[np.ndarray]):
        self.key = key                    # 文件名（唯一标识）
        self.filename = filename
        self.matted_path = matted_path    # 批量抠图结果（基准来源之一）
        self.orig_path = orig_path        # 未抠图原始帧（还原橡皮图源）
        self.refined_path = refined_path  # 精抠输出路径
        self.base = base                  # 进入精抠时的基准 BGRA
        self.orig = orig                  # 原始帧 BGRA（可能为 None）
        self.work = base.copy()           # 当前工作图 BGRA
        self.undo: List[Tuple[int, int, int, int, np.ndarray]] = []
        self.modified = False             # 相对已落盘内容是否有改动

        # 一笔进行中的撤销信息：bbox + 修改前补丁
        self._bbox: Optional[List[int]] = None
        self._before: Optional[np.ndarray] = None

    # ----- 笔画级撤销记录 -----

    def stroke_begin(self):
        self._bbox = None
        self._before = None

    def stroke_snapshot(self, work: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        """在修改该圆形区域前调用：把"修改前"像素并入本笔快照"""
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(work.shape[1], x1), min(work.shape[0], y1)
        if x0 >= x1 or y0 >= y1:
            return
        if self._bbox is None:
            self._bbox = [x0, y0, x1, y1]
            self._before = work[y0:y1, x0:x1].copy()
            return
        bx0, by0, bx1, by1 = self._bbox
        ux0, uy0, ux1, uy1 = min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1)
        if (ux0, uy0, ux1, uy1) == (bx0, by0, bx1, by1):
            return
        # 先取修改前的并集，再把旧补丁贴回对应位置
        merged = work[uy0:uy1, ux0:ux1].copy()
        merged[by0 - uy0:by1 - uy0, bx0 - ux0:bx1 - ux0] = self._before
        self._before = merged
        self._bbox = [ux0, uy0, ux1, uy1]

    def stroke_commit(self):
        """一笔结束：压入撤销栈"""
        if self._bbox is not None and self._before is not None:
            self.undo.append((self._bbox[0], self._bbox[1],
                              self._bbox[2], self._bbox[3], self._before))
            if len(self.undo) > self.UNDO_LIMIT:
                self.undo.pop(0)
            self.modified = True
        self._bbox = None
        self._before = None

    def stroke_cancel(self):
        self._bbox = None
        self._before = None

    def apply_undo(self, work: np.ndarray) -> bool:
        if not self.undo:
            return False
        x0, y0, x1, y1, patch = self.undo.pop()
        work[y0:y1, x0:x1] = patch
        self.modified = True
        return True


# ===================== 可涂抹画布 =====================

class RefineCanvas(QWidget):
    """显示工作图（棋盘格透明底），接收鼠标涂抹；支持滚轮缩放/拖拽平移"""

    TOOL_COLORS = {
        "erase": QColor(231, 76, 60),       # 红
        "bg": QColor(241, 196, 15),          # 黄
        "restore": QColor(46, 204, 113),    # 绿
    }

    ZOOM_MIN = 1.0      # 1 = 适应窗口
    ZOOM_MAX = 24.0

    def __init__(self, page: "FrameRefinePage"):
        super().__init__()
        self.page = page
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background:#2b2b2b;border:1px solid #444;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._src_pixmap: Optional[QPixmap] = None  # 全尺寸工作图
        self._hover: Optional[QPointF] = None       # 原图坐标悬停点
        self._drawing = False

        # 视图变换：zoom=1 为适应窗口；center_img 为控件中心对应的原图坐标
        self._zoom = 1.0
        self._center_img: Optional[QPointF] = None
        self._fast = False                 # 交互中用快速缩放

        # 平移
        self._panning = False
        self._pan_start_widget = QPointF()
        self._pan_start_center = QPointF()
        self._space_held = False

    # ---------- 坐标映射 ----------

    def image_size(self) -> Optional[Tuple[int, int]]:
        return self.page.image_size()

    def _fit_scale(self) -> Optional[float]:
        sz = self.image_size()
        if sz is None:
            return None
        iw, ih = sz
        if iw <= 0 or ih <= 0:
            return None
        return min(self.width() / iw, self.height() / ih)

    def _default_center(self, iw: float, ih: float) -> QPointF:
        return QPointF(iw / 2.0, ih / 2.0)

    def _clamp_center(self, c: QPointF, s: float,
                      iw: float, ih: float) -> QPointF:
        """限制视图中心，保证画面不会被拖到完全看不见"""
        mx = (self.width() / 2.0) / s
        my = (self.height() / 2.0) / s
        cx = min(max(c.x(), -mx), iw + mx)
        cy = min(max(c.y(), -my), ih + my)
        return QPointF(cx, cy)

    def _transform(self) -> Optional[Tuple[float, float, float, QPointF]]:
        """返回 scale, offset_x, offset_y, center_img（控件坐标映射）"""
        sz = self.image_size()
        fit = self._fit_scale()
        if sz is None or fit is None:
            return None
        iw, ih = sz
        s = fit * self._zoom
        if self._center_img is None:
            self._center_img = self._default_center(iw, ih)
        c = self._clamp_center(self._center_img, s, iw, ih)
        self._center_img = c
        cw = self.width() / 2.0
        ch = self.height() / 2.0
        ox = cw - c.x() * s
        oy = ch - c.y() * s
        return s, ox, oy, c

    def widget_to_image(self, pos: QPointF) -> Optional[Tuple[float, float]]:
        t = self._transform()
        if t is None:
            return None
        s, ox, oy, _ = t
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    # ---------- 场景刷新 ----------

    def rebuild_scene(self, smooth: bool = True):
        """根据 page.work 更新全尺寸像素图（保持当前缩放/平移）"""
        self._fast = not smooth
        work = self.page.work
        if work is None:
            self._src_pixmap = None
            self.update()
            return
        rgba = cv2.cvtColor(work, cv2.COLOR_BGRA2RGBA)
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self._src_pixmap = QPixmap.fromImage(qimg)
        if self._center_img is None:
            self._center_img = self._default_center(w, h)
        self.update()

    def reset_view(self):
        """切换图片时：缩放复位到适应窗口、居中"""
        self._zoom = 1.0
        sz = self.image_size()
        self._center_img = (self._default_center(*sz)
                            if sz is not None else None)
        self._hover = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(43, 43, 43))

        if self._src_pixmap is None:
            p.setPen(QColor("#bdc3c7"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "没有可精抠的图片\n\n请先在「批量抠图」页面处理选中帧")
        else:
            t = self._transform()
            if t is not None:
                s, ox, oy, _ = t
                iw = self._src_pixmap.width()
                ih = self._src_pixmap.height()
                target = QRectF(ox, oy, iw * s, ih * s)
                p.setRenderHint(QPainter.SmoothPixmapTransform, not self._fast)
                # 棋盘格只铺在图片区域
                p.save()
                p.setClipRect(target)
                p.drawTiledPixmap(target, _checker_tile(12))
                p.drawPixmap(target, self._src_pixmap,
                             QRectF(0, 0, iw, ih))
                p.restore()

        # 画笔圈
        if self._hover is not None and self.page.work is not None:
            t = self._transform()
            if t is not None:
                s, ox, oy, _ = t
                radius = max(1.0, self.page.brush_size() * s)
                cx = self._hover.x() * s + ox
                cy = self._hover.y() * s + oy
                color = self.TOOL_COLORS.get(self.page.tool(), QColor("white"))
                p.setPen(QPen(color, 1.6))
                p.drawEllipse(QPointF(cx, cy), radius, radius)
                p.setPen(QPen(QColor(0, 0, 0, 140), 0.8))
                p.drawEllipse(QPointF(cx + 1, cy + 1), radius, radius)
        p.end()

        # 缩放比例角标
        if self.page.work is not None:
            p2 = QPainter(self)
            txt = f"{self._zoom * 100:.0f}%"
            font = p2.font()
            font.setPointSize(9)
            p2.setFont(font)
            tw = p2.fontMetrics().horizontalAdvance(txt)
            r = QRect(8, 8, tw + 14, 20)
            p2.setPen(Qt.NoPen)
            p2.setBrush(QColor(0, 0, 0, 130))
            p2.drawRoundedRect(r, 4, 4)
            p2.setPen(QColor("#ecf0f1"))
            p2.drawText(r, Qt.AlignCenter, txt)
            p2.end()

    def resizeEvent(self, event):
        self.update()

    def leaveEvent(self, event):
        self._hover = None
        self.update()

    # ---------- 滚轮缩放（以鼠标位置为锚点） ----------

    def wheelEvent(self, event):
        if self.page.work is None:
            return
        t = self._transform()
        fit = self._fit_scale()
        if t is None or fit is None:
            return
        mouse = event.position()
        anchor = self.widget_to_image(mouse)  # 缩放前鼠标下的原图点

        factor = 1.2 ** (event.angleDelta().y() / 120.0)
        new_zoom = min(max(self._zoom * factor, self.ZOOM_MIN), self.ZOOM_MAX)
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        self._zoom = new_zoom
        new_s = fit * new_zoom

        # 保持锚点像素不动：center = anchor - (mouse - widgetCenter)/new_s
        cw = self.width() / 2.0
        ch = self.height() / 2.0
        self._center_img = QPointF(
            anchor[0] - (mouse.x() - cw) / new_s,
            anchor[1] - (mouse.y() - ch) / new_s)
        self._fast = False
        self.update()

    # ---------- 双击复位 ----------

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.page.work is not None:
            # 双击前会先产生一次"按下-抬起"，若没有拖动，撤销那枚误盖笔印
            if not getattr(self.page, "_stroke_moved", True):
                self.page._undo()
            self.reset_view()

    # ---------- 空格键状态（空格+左键 = 平移） ----------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.unsetCursor()
        else:
            super().keyReleaseEvent(event)

    # ---------- 鼠标涂抹 / 平移 ----------

    def mousePressEvent(self, event):
        if self.page.work is None:
            return
        # 中键，或按住空格的左键 → 平移
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and self._space_held):
            self._panning = True
            self._pan_start_widget = event.position()
            t = self._transform()
            self._pan_start_center = t[3] if t else QPointF()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            return
        ip = self.widget_to_image(event.position())
        if ip is None:
            return
        self._drawing = True
        self._hover = QPointF(*ip)
        self.page.stroke_start(ip)

    def mouseMoveEvent(self, event):
        if self._panning:
            t = self._transform()
            if t is not None:
                s = t[0]
                d = event.position() - self._pan_start_widget
                self._center_img = QPointF(
                    self._pan_start_center.x() - d.x() / s,
                    self._pan_start_center.y() - d.y() / s)
                self._fast = True
                self.update()
            return
        ip = self.widget_to_image(event.position())
        if ip is None:
            return
        self._hover = QPointF(*ip)
        if self._drawing:
            self.page.stroke_apply(ip, smooth=False)
        else:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and self._panning):
            self._panning = False
            self._fast = False
            self.setCursor(Qt.OpenHandCursor if self._space_held
                           else Qt.ArrowCursor)
            self.update()
            return
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self.page.stroke_end()


# ===================== 缩略图单元格 =====================

class RefineCell(QFrame):
    """精抠图片缩略图：蓝框=当前编辑，绿色标题=已精修，●=有未保存修改"""

    clicked = Signal(str)

    def __init__(self, key: str, thumb_path: str,
                 thumb_w: int = 110, thumb_h: int = 80):
        super().__init__()
        self.key = key
        self._current = False
        self._refined = False
        self._dirty = False

        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(thumb_w + 14)
        self.setFixedHeight(thumb_h + 34)
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(thumb_w, thumb_h)
        self._thumb_w, self._thumb_h = thumb_w, thumb_h
        lay.addWidget(self.img_label, 0, Qt.AlignHCenter)

        self.caption = QLabel()
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setFixedHeight(16)
        lay.addWidget(self.caption, 0, Qt.AlignHCenter)

        self.set_thumb(thumb_path)
        self._update_style()

    def set_thumb(self, path: str):
        src = QPixmap(path)
        canvas = QPixmap(self._thumb_w, self._thumb_h)
        p = QPainter(canvas)
        p.drawTiledPixmap(canvas.rect(), _checker_tile(10))
        if not src.isNull():
            scaled = src.scaled(self._thumb_w, self._thumb_h,
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap((self._thumb_w - scaled.width()) // 2,
                         (self._thumb_h - scaled.height()) // 2, scaled)
        p.end()
        self.img_label.setPixmap(canvas)

    def set_state(self, current: bool, refined: bool, dirty: bool,
                 number: int):
        self._current, self._refined, self._dirty = current, refined, dirty
        mark = "● " if dirty else ""
        self.caption.setText(f"{mark}#{number}")
        self._update_style()

    def _update_style(self):
        if self._current:
            border, bg, fg = "#f1c40f", "#1a3a5c", "#f1c40f"
        elif self._refined:
            border, bg, fg = "#2ecc71", "#14301f", "#2ecc71"
        else:
            border, bg, fg = "#444444", "#1e1e1e", "#888888"
        self.setStyleSheet(
            f"RefineCell{{background:{bg};border:3px solid {border};"
            f"border-radius:4px;}}"
            f"QLabel{{background:transparent;color:{fg};font-size:12px;}}")

    def mousePressEvent(self, event):
        self.clicked.emit(self.key)


# ===================== 精抠页面 =====================

class FrameRefinePage(QWidget):
    """图片精抠页：人工修正批量抠图结果，每图独立撤销"""

    def __init__(self, shared_state: dict):
        super().__init__()
        self.shared_state = shared_state
        self.setStyleSheet("QWidget{background:#2b2b2b;color:#ecf0f1;}")

        # 数据
        self.matted_dir: str = ""
        self.frames_dir: str = ""
        self.refined_dir: str = ""
        self.items: List[dict] = []                 # 有序图片项
        self.states: Dict[str, EditState] = {}      # key=文件名
        self.cells: List[RefineCell] = []
        self.cur_key: Optional[str] = None
        self.work: Optional[np.ndarray] = None      # 当前工作图（=state.work 引用）
        self._last_pt: Optional[Tuple[float, float]] = None
        self._stroke_moved = False
        self._file_sig: str = ""

        # 工具参数
        self._tool = "erase"

        # resize 防抖
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._build_grid)

        self._build_ui()
        self._bind_shortcuts()

    # ---------- UI ----------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ===== 左侧工作区 =====
        work_box = QGroupBox("精抠工作区（棋盘格=透明）")
        wl = QVBoxLayout(work_box)

        self.canvas = RefineCanvas(self)
        wl.addWidget(self.canvas, 1)

        root.addWidget(work_box, 13)

        # ===== 右侧面板 =====
        right_panel = QWidget()
        rlay = QVBoxLayout(right_panel)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(8)

        # --- 图片导航（从左侧工作区移入） ---
        nav_box = QGroupBox("图片导航")
        nv = QVBoxLayout(nav_box)
        nv.setSpacing(4)
        nav_row = QHBoxLayout()
        btn_prev = QPushButton("⏮ 上一张")
        btn_next = QPushButton("下一张 ⏭")
        btn_prev.clicked.connect(self._prev_image)
        btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(btn_prev)
        nav_row.addWidget(btn_next)
        nav_row.addStretch()
        nv.addLayout(nav_row)
        pos_row = QHBoxLayout()
        self.pos_label = QLabel("- / -")
        self.pos_label.setStyleSheet("color:#888;")
        pos_row.addWidget(self.pos_label)
        pos_row.addStretch()
        self.state_label = QLabel("")
        self.state_label.setStyleSheet("color:#2ecc71;")
        pos_row.addWidget(self.state_label)
        nv.addLayout(pos_row)
        rlay.addWidget(nav_box, 0)

        # --- 数据源/输出 ---
        dir_box = QGroupBox("图片来源")
        dl = QVBoxLayout(dir_box)
        dl.setSpacing(3)
        self.src_label = QLabel("批量抠图结果：（未加载）")
        self.src_label.setWordWrap(True)
        self.src_label.setStyleSheet("color:#aaa;font-size:12px;")
        dl.addWidget(self.src_label)
        self.out_label = QLabel("精抠输出：（未加载）")
        self.out_label.setWordWrap(True)
        self.out_label.setStyleSheet("color:#aaa;font-size:12px;")
        dl.addWidget(self.out_label)
        btn_row = QHBoxLayout()
        btn_reload = QPushButton("重新加载")
        btn_reload.clicked.connect(self._manual_reload)
        btn_open = QPushButton("打开输出目录")
        btn_open.clicked.connect(self._open_out_dir)
        btn_row.addWidget(btn_reload)
        btn_row.addWidget(btn_open)
        dl.addLayout(btn_row)
        rlay.addWidget(dir_box, 0)

        # --- 工具 ---
        tool_box = QGroupBox("工具")
        tl = QVBoxLayout(tool_box)
        tl.setSpacing(4)
        tool_row = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        self.rb_erase = QRadioButton("橡皮擦")
        self.rb_bg = QRadioButton("背景橡皮擦")
        self.rb_restore = QRadioButton("还原橡皮擦")
        self.rb_erase.setChecked(True)
        for rb in (self.rb_erase, self.rb_bg, self.rb_restore):
            self.tool_group.addButton(rb)
            tool_row.addWidget(rb)
        self.rb_erase.toggled.connect(self._on_tool_changed)
        self.rb_bg.toggled.connect(self._on_tool_changed)
        self.rb_restore.toggled.connect(self._on_tool_changed)
        tl.addLayout(tool_row)
        tool_tip = QLabel(
            "橡皮擦：擦掉不要的部分\n"
            "背景橡皮擦：只擦与落笔处颜色相近的像素，保护主体\n"
            "还原橡皮擦：从原始帧补回被误擦的部分")
        tool_tip.setStyleSheet("color:#888;font-size:11px;")
        tl.addWidget(tool_tip)
        rlay.addWidget(tool_box, 0)

        # --- 笔刷参数 ---
        brush_box = QGroupBox("笔刷")
        bl = QVBoxLayout(brush_box)
        bl.setSpacing(4)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("大小:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(2, 150)
        self.size_slider.setValue(30)
        self.size_label = QLabel("30")
        self.size_label.setMinimumWidth(30)
        self.size_slider.valueChanged.connect(
            lambda v: self.size_label.setText(str(v)))
        size_row.addWidget(self.size_slider, 1)
        size_row.addWidget(self.size_label)
        bl.addLayout(size_row)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("背景橡皮容差:"))
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(5, 150)
        self.tol_slider.setValue(40)
        self.tol_label = QLabel("40")
        self.tol_label.setMinimumWidth(30)
        self.tol_slider.valueChanged.connect(
            lambda v: self.tol_label.setText(str(v)))
        tol_row.addWidget(self.tol_slider, 1)
        tol_row.addWidget(self.tol_label)
        bl.addLayout(tol_row)
        rlay.addWidget(brush_box, 0)

        # --- 操作 ---
        op_box = QGroupBox("操作（每张图独立撤销）")
        ol = QVBoxLayout(op_box)
        ol.setSpacing(4)
        self.btn_undo = QPushButton("↩ 撤销（Ctrl+Z）")
        self.btn_undo.clicked.connect(self._undo)
        ol.addWidget(self.btn_undo)
        self.btn_revert = QPushButton("整张贴回批量抠图结果")
        self.btn_revert.setToolTip("放弃本张所有精抠修改（本操作本身也可撤销）")
        self.btn_revert.clicked.connect(self._revert_image)
        ol.addWidget(self.btn_revert)
        save_row = QHBoxLayout()
        self.btn_save = QPushButton("💾 保存当前")
        self.btn_save.clicked.connect(self._save_current)
        self.btn_save_all = QPushButton("全部保存")
        self.btn_save_all.clicked.connect(self._save_all)
        save_row.addWidget(self.btn_save)
        save_row.addWidget(self.btn_save_all)
        ol.addLayout(save_row)
        rlay.addWidget(op_box, 0)

        # 操作提示（从左侧工作区底部移入）
        hint = QLabel("提示：左键拖动涂抹；滚轮缩放（以光标为中心）；中键或空格+左键拖动平移；双击适应窗口；Ctrl+Z 撤销；[ ] 调笔刷")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7f8c8d;font-size:11px;")
        rlay.addWidget(hint, 0)
        rlay.addStretch(1)

        # --- 下半：精抠图片网格 ---
        self.grid_group = QGroupBox("精抠图片")
        gl = QVBoxLayout(self.grid_group)
        gl.setContentsMargins(8, 8, 8, 8)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_host = QWidget()
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid_scroll.setWidget(self.grid_host)
        gl.addWidget(self.grid_scroll)
        rlay2 = QVBoxLayout()
        rlay2.setContentsMargins(0, 0, 0, 0)
        rlay2.addWidget(right_panel, 1)
        rlay2.addWidget(self.grid_group, 1)
        wrap = QWidget()
        wrap.setLayout(rlay2)
        wrap_scroll = QScrollArea()
        wrap_scroll.setWidgetResizable(True)
        wrap_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrap_scroll.setStyleSheet("QScrollArea{background:#2b2b2b;border:none;}")
        wrap_scroll.setWidget(wrap)
        root.addWidget(wrap_scroll, 8)

        self._update_ops()

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("["), self, lambda: self._adjust_brush(-4))
        QShortcut(QKeySequence("]"), self, lambda: self._adjust_brush(4))

    # ---------- 供画布查询的状态 ----------

    def image_size(self):
        if self.work is None:
            return None
        return self.work.shape[1], self.work.shape[0]

    def tool(self) -> str:
        return self._tool

    def brush_size(self) -> int:
        return self.size_slider.value()

    # ---------- 加载 / 刷新 ----------

    def showEvent(self, event):
        super().showEvent(event)
        self._reload_if_changed()

    def _manual_reload(self):
        self._file_sig = ""  # 强制重载
        self._reload_if_changed(force=True)

    def _reload_if_changed(self, force: bool = False):
        matted_dir = self.shared_state.get("matted_dir", "")
        frames_dir = self.shared_state.get("frames_dir", "")
        if not matted_dir or not os.path.isdir(matted_dir):
            self._show_empty("尚未找到批量抠图结果\n\n请先在「批量抠图」页面处理选中帧")
            return
        files = sorted(f for f in os.listdir(matted_dir)
                       if f.lower().endswith(".png"))
        sig = matted_dir + "|" + "|".join(files)
        if not force and sig == self._file_sig:
            return
        self._file_sig = sig

        # 切换图片前先保存当前修改
        self._flush_current(show_msg=False)

        self.matted_dir = matted_dir
        self.frames_dir = frames_dir
        self.refined_dir = os.path.join(
            frames_dir, "refined") if frames_dir else os.path.join(
                os.path.dirname(matted_dir), "refined")
        os.makedirs(self.refined_dir, exist_ok=True)

        old_states = self.states
        self.items = []
        self.states = {}
        for i, fn in enumerate(files):
            stem = os.path.splitext(fn)[0]
            matted_path = os.path.join(matted_dir, fn)
            refined_path = os.path.join(self.refined_dir, fn)
            orig_path = self._find_original(stem)
            self.items.append({
                "key": fn, "number": i + 1,
                "matted": matted_path, "refined": refined_path,
                "orig": orig_path,
            })

        self.src_label.setText(f"批量抠图结果：{matted_dir}")
        self.out_label.setText(f"精抠输出：{self.refined_dir}")

        # 只保留未保存的内存编辑；已落盘的重新从磁盘加载（撤销栈重置）
        self.cur_key = self.cur_key if any(
            it["key"] == self.cur_key for it in self.items) else None
        self.states = {k: v for k, v in old_states.items()
                       if v.modified
                       and any(it["key"] == k for it in self.items)}

        self._build_grid()
        if self.cur_key is None and self.items:
            self.cur_key = self.items[0]["key"]
        self._open_current()
        n_refined = sum(1 for it in self.items
                        if os.path.isfile(it["refined"]))
        self.status_text(f"已加载 {len(self.items)} 张，已精修 {n_refined} 张")

    def _find_original(self, stem: str) -> Optional[str]:
        """按文件 stem 在原始帧目录找未抠图原图"""
        if not self.frames_dir or not os.path.isdir(self.frames_dir):
            return None
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            p = os.path.join(self.frames_dir, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def _show_empty(self, text: str):
        self.items, self.states, self.cells = [], {}, []
        self.cur_key = None
        self.work = None
        self._file_sig = ""
        self.matted_dir = ""
        self.refined_dir = ""
        self.canvas.rebuild_scene()
        self.canvas.reset_view()
        self._build_grid()
        self.pos_label.setText("- / -")
        self.state_label.setText("")
        self.src_label.setText(text.replace("\n", " "))
        self.out_label.setText("精抠输出：（未加载）")
        self._update_ops()

    def status_text(self, text: str):
        self.state_label.setText(text)

    # ---------- 状态惰性加载 ----------

    def _get_state(self, item: dict) -> Optional[EditState]:
        key = item["key"]
        st = self.states.get(key)
        if st is not None:
            return st
        # 优先读取已精修结果，其次批量抠图结果
        base_path = item["refined"] if os.path.isfile(item["refined"]) \
            else item["matted"]
        base = imread_unicode(base_path)
        if base is None:
            return None
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGRA)
        elif base.shape[2] == 3:
            base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
        orig = None
        if item["orig"]:
            o = imread_unicode(item["orig"])
            if o is not None:
                if o.ndim == 2:
                    o = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
                if o.shape[2] == 3:
                    o = cv2.cvtColor(o, cv2.COLOR_BGR2BGRA)
                if o.shape[:2] == base.shape[:2]:
                    orig = o
        st = EditState(key, item["key"], item["matted"], item["orig"],
                       item["refined"], base, orig)
        self.states[key] = st
        return st

    # ---------- 图片切换 ----------

    def _cur_item(self) -> Optional[dict]:
        for it in self.items:
            if it["key"] == self.cur_key:
                return it
        return None

    def _open_current(self):
        it = self._cur_item()
        if it is None:
            self.work = None
            self.canvas.rebuild_scene()
            self.pos_label.setText("- / -")
            self._update_ops()
            return
        st = self._get_state(it)
        if st is None:
            self.work = None
            self.canvas.rebuild_scene()
            return
        self.work = st.work
        self.canvas.rebuild_scene(smooth=True)
        self.canvas.reset_view()
        idx = self.items.index(it) + 1
        self.pos_label.setText(f"{idx} / {len(self.items)}  {it['key']}")
        self._refresh_cells()
        self._update_ops()

    def _flush_current(self, show_msg: bool = True):
        """把当前编辑落盘到 refined/"""
        if self.cur_key is None:
            return
        st = self.states.get(self.cur_key)
        if st is None or not st.modified:
            return
        os.makedirs(self.refined_dir, exist_ok=True)
        if imwrite_unicode(st.refined_path, st.work):
            st.modified = False
            self._write_info()
            self.shared_state["refined_dir"] = self.refined_dir
            self._refresh_cells()
            self._update_ops()
            if show_msg:
                self.status_text(f"已保存：{os.path.basename(st.refined_path)}")

    def _switch_to(self, key: str):
        if key == self.cur_key:
            return
        self._flush_current(show_msg=False)
        self.cur_key = key
        self._open_current()

    def _prev_image(self):
        if not self.items:
            return
        idx = next((i for i, it in enumerate(self.items)
                    if it["key"] == self.cur_key), 0)
        self._switch_to(self.items[(idx - 1) % len(self.items)]["key"])

    def _next_image(self):
        if not self.items:
            return
        idx = next((i for i, it in enumerate(self.items)
                    if it["key"] == self.cur_key), 0)
        self._switch_to(self.items[(idx + 1) % len(self.items)]["key"])

    # ---------- 工具 / 参数 ----------

    def _on_tool_changed(self):
        if self.rb_erase.isChecked():
            self._tool = "erase"
        elif self.rb_bg.isChecked():
            self._tool = "bg"
        else:
            self._tool = "restore"
        self.canvas.update()

    def _adjust_brush(self, delta: int):
        self.size_slider.setValue(
            max(self.size_slider.minimum(),
                min(self.size_slider.maximum(),
                    self.size_slider.value() + delta)))

    # ---------- 笔画核心 ----------

    def stroke_start(self, pt: Tuple[float, float]):
        st = self._cur_state()
        if st is None:
            return
        st.stroke_begin()
        self._stroke_moved = False
        self._last_pt = pt
        self._apply_at(pt)

    def stroke_apply(self, pt: Tuple[float, float], smooth: bool = False):
        self._stroke_moved = True
        if self._last_pt is None:
            self._apply_at(pt)
        else:
            # 移动点之间插值，避免快速拖动出现断笔
            x0, y0 = self._last_pt
            x1, y1 = pt
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            step = max(1.0, self.brush_size() / 4.0)
            n = max(1, int(dist / step))
            for i in range(1, n + 1):
                self._apply_at((x0 + (x1 - x0) * i / n,
                                y0 + (y1 - y0) * i / n))
        self._last_pt = pt
        self.canvas.rebuild_scene(smooth=smooth)

    def stroke_end(self):
        st = self._cur_state()
        if st is not None:
            st.stroke_commit()
            self._after_edit()
        self._last_pt = None
        self.canvas.rebuild_scene(smooth=True)

    def _cur_state(self) -> Optional[EditState]:
        if self.cur_key is None:
            return None
        return self.states.get(self.cur_key)

    def _apply_at(self, pt: Tuple[float, float]):
        st = self._cur_state()
        if st is None or self.work is None:
            return
        radius = self.brush_size() // 2
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius + 1, cy + radius + 1
        h, w = self.work.shape[:2]
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x1), min(h, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            return

        # 修改前先并入撤销快照
        st.stroke_snapshot(self.work, x0, y0, x1, y1)

        yy, xx = np.ogrid[cy0:cy1, cx0:cx1]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        roi = self.work[cy0:cy1, cx0:cx1]

        if self._tool == "erase":
            roi[circle, 3] = 0

        elif self._tool == "bg":
            # 连续取样：落笔点当前颜色作为背景色，圆内相近色降 alpha
            if 0 <= cy < h and 0 <= cx < w:
                sample = self.work[cy, cx, :3].astype(np.int16)
                diff = roi[:, :, :3].astype(np.int16) - sample
                dist = np.linalg.norm(diff, axis=2)
                tol = float(self.tol_slider.value())
                inner = tol * 0.6
                keep = np.clip((dist - inner) / max(tol - inner, 1e-6),
                               0.0, 1.0)
                new_a = (roi[:, :, 3].astype(np.float32) * keep)
                m = circle & (dist < tol)
                roi[m, 3] = new_a[m].astype(np.uint8)

        elif self._tool == "restore":
            if st.orig is None:
                self.status_text("找不到原始帧，无法还原（请确认 frames 目录存在）")
                return
            orig_roi = st.orig[cy0:cy1, cx0:cx1]
            m3 = np.repeat(circle[:, :, None], 4, axis=2)
            roi[m3] = orig_roi[m3]

    def _after_edit(self):
        self._refresh_cells()
        self._update_ops()

    # ---------- 撤销 / 整张贴回 ----------

    def _undo(self):
        st = self._cur_state()
        if st is None:
            return
        if st.apply_undo(st.work):
            self.work = st.work
            self.canvas.rebuild_scene()
            self._after_edit()
            self.status_text(f"已撤销（剩余可撤销 {len(st.undo)} 步）")
        else:
            self.status_text("本张图片没有可撤销的操作")

    def _revert_image(self):
        st = self._cur_state()
        if st is None or self.work is None:
            return
        ret = QMessageBox.question(
            self, "确认",
            "把本张贴回批量抠图结果？\n（当前精抠内容会被覆盖，此操作可用撤销找回）")
        if ret != QMessageBox.Yes:
            return
        # 整图快照入栈，保证可撤销
        st.stroke_begin()
        st.stroke_snapshot(st.work, 0, 0, st.work.shape[1], st.work.shape[0])
        st.work[:] = st.base
        st.stroke_commit()
        self.work = st.work
        self.canvas.rebuild_scene()
        self._after_edit()
        self.status_text("已贴回批量抠图结果（可撤销）")

    # ---------- 保存 ----------

    def _save_current(self):
        self._flush_current(show_msg=True)

    def _save_all(self):
        self._flush_current(show_msg=False)
        n = 0
        os.makedirs(self.refined_dir, exist_ok=True)
        for it in self.items:
            st = self.states.get(it["key"])
            if st is not None and st.modified:
                if imwrite_unicode(st.refined_path, st.work):
                    st.modified = False
                    n += 1
        self._write_info()
        self.shared_state["refined_dir"] = self.refined_dir
        self._refresh_cells()
        self._update_ops()
        self.status_text(f"全部保存完成（本次写入 {n} 张）→ {self.refined_dir}")

    def _write_info(self):
        data = {
            "matted_dir": os.path.abspath(self.matted_dir),
            "frames_dir": os.path.abspath(self.frames_dir) if self.frames_dir else "",
            "output_dir": os.path.abspath(self.refined_dir),
            "count": len(self.items),
            "frames": [
                {"index": it["number"], "filename": it["key"]}
                for it in self.items
                if os.path.isfile(it["refined"])
            ],
        }
        try:
            with open(os.path.join(self.refined_dir, "refined_info.json"),
                      "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _open_out_dir(self):
        if not self.refined_dir or not os.path.isdir(self.refined_dir):
            QMessageBox.information(self, "提示", "输出目录不存在")
            return
        os.startfile(self.refined_dir)

    # ---------- 网格 ----------

    def _build_grid(self):
        while self.grid_layout.count():
            it = self.grid_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.cells = []
        if not self.items:
            return
        avail = self.grid_scroll.viewport().width() - 20
        cell_w = 124 + 6
        cols = max(1, avail // cell_w)
        for i, item in enumerate(self.items):
            path = item["refined"] if os.path.isfile(item["refined"]) \
                else item["matted"]
            cell = RefineCell(item["key"], path)
            cell.clicked.connect(self._switch_to)
            self.grid_layout.addWidget(cell, i // cols, i % cols)
            self.cells.append(cell)
        self._refresh_cells()

    def _refresh_cells(self):
        refined_keys = {it["key"] for it in self.items
                        if os.path.isfile(it["refined"])}
        for i, item in enumerate(self.items):
            if i >= len(self.cells):
                continue
            cell = self.cells[i]
            st = self.states.get(item["key"])
            dirty = bool(st and st.modified)
            cell.set_state(current=item["key"] == self.cur_key,
                           refined=item["key"] in refined_keys,
                           dirty=dirty, number=item["number"])

    def _update_ops(self):
        st = self._cur_state()
        has = st is not None and self.work is not None
        n_undo = len(st.undo) if st else 0
        self.btn_undo.setEnabled(n_undo > 0)
        self.btn_undo.setText(f"↩ 撤销（Ctrl+Z）  [{n_undo}]")
        self.btn_revert.setEnabled(has)
        self.btn_save.setEnabled(has)
        self.btn_save_all.setEnabled(any(
            s.modified for s in self.states.values()))
        if st:
            if st.modified:
                self.state_label.setText("● 有未保存修改")
                self.state_label.setStyleSheet("color:#f39c12;")
            elif os.path.isfile(st.refined_path):
                self.state_label.setText("✓ 已精修保存")
                self.state_label.setStyleSheet("color:#2ecc71;")
            else:
                self.state_label.setText("未修改")
                self.state_label.setStyleSheet("color:#888;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.items:
            self._resize_timer.start()

