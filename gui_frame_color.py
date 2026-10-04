# -*- coding: utf-8 -*-
"""
模块五：图片色调统一 - 页面（可嵌入主窗口）

以一张「基板图」的主体颜色为基准，把所有抠好的图片统一到同一色调，
消除序列帧之间的颜色闪烁。提供两种算法：
  - Reinhard 颜色迁移：在 Lab 空间对齐均值/标准差，色调自然（默认）
  - 直方图匹配：逐通道匹配累积分布，对齐更强力

统计与变换只作用于主体像素（按 alpha 加权），背景色不参与、不被污染；
alpha 通道原样保留。输入优先取精抠结果（refined/），无则取批量抠图
（matted/）；输出到 frames_dir/colored/ 同名 PNG。

布局与其它页面一致：左右固定 13:8。左侧上下两部分——
上=待矫色的抠图（棋盘格透明底），下=基板原图（颜色参考）；
右侧上面功能按钮，下面全部图片缩略图网格。
"""

import os
import json
import shutil
from typing import Optional, List, Dict

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSlider, QScrollArea, QGroupBox, QRadioButton, QButtonGroup,
    QCheckBox, QFrame, QSplitter, QFileDialog, QMessageBox,
)

from gui_frame_matting import (
    imread_unicode, imwrite_unicode, compose_checkerboard,
)


# ===================== 色调统计算法 =====================

def _lab_stats(bgr: np.ndarray, w: np.ndarray):
    """加权像素在 Lab 空间的均值/标准差（w 形状与图像相同，取值 0~1）"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ws = w.sum()
    if ws < 1e-6:
        return lab, np.zeros(3, np.float32), np.ones(3, np.float32)
    mean = (lab * w[..., None]).reshape(-1, 3).sum(0) / ws
    var = (((lab - mean) ** 2) * w[..., None]).reshape(-1, 3).sum(0) / ws
    return lab, mean, np.sqrt(np.maximum(var, 1e-6))


def reinhard_transfer(src_bgra: np.ndarray, ref_bgra: np.ndarray,
                      strength: float) -> np.ndarray:
    """
    Reinhard 颜色迁移（Lab 空间）：把源图主体的均值/标准差对齐到参考图。
    统计按 alpha 加权（只用主体像素），变换施加于 alpha>0 的像素。
    strength: 0~1，结果 = 源图与矫色结果按比例混合。
    """
    src_bgr = src_bgra[:, :, :3]
    ref_bgr = ref_bgra[:, :, :3]
    sa = src_bgra[:, :, 3].astype(np.float32) / 255.0
    ra = ref_bgra[:, :, 3].astype(np.float32) / 255.0
    mask = sa > 0
    if not mask.any() or not (ra > 0).any():
        return src_bgra.copy()

    lab_s, m_s, s_s = _lab_stats(src_bgr, sa)
    _, m_r, s_r = _lab_stats(ref_bgr, ra)
    ratio = s_r / np.maximum(s_s, 1e-3)
    corrected_lab = (lab_s - m_s) * ratio + m_r
    corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    out = src_bgra.copy()
    if strength >= 0.999:
        out[mask, :3] = corrected[mask]
    else:
        mix = (src_bgr.astype(np.float32) * (1 - strength)
               + corrected.astype(np.float32) * strength)
        out[mask, :3] = np.clip(mix[mask], 0, 255).astype(np.uint8)
    return out


def _weighted_lut(src_c: np.ndarray, ref_c: np.ndarray,
                  ws: np.ndarray, wr: np.ndarray) -> np.ndarray:
    """按加权直方图的累积分布，构造 src 0~255 → ref 的匹配查找表"""
    hs = np.bincount(src_c, weights=ws, minlength=256)
    hr = np.bincount(ref_c, weights=wr, minlength=256)
    cs = np.cumsum(hs)
    cr = np.cumsum(hr)
    if cs[-1] <= 0 or cr[-1] <= 0:
        return np.arange(256, dtype=np.uint8)
    cs /= cs[-1]
    cr /= cr[-1]
    # 每个源灰度找累积分布最接近的参考灰度
    idx = np.searchsorted(cr, cs)
    lut = np.clip(idx, 0, 255).astype(np.uint8)
    return lut


def hist_match_transfer(src_bgra: np.ndarray, ref_bgra: np.ndarray,
                        strength: float) -> np.ndarray:
    """
    直方图匹配（逐 BGR 通道）：让源图主体的直方图分布对齐参考图。
    直方图按 alpha 加权统计，匹配施加于 alpha>0 的像素。
    """
    src_bgr = src_bgra[:, :, :3]
    ref_bgr = ref_bgra[:, :, :3]
    sa = src_bgra[:, :, 3].astype(np.float32) / 255.0
    ra = ref_bgra[:, :, 3].astype(np.float32) / 255.0
    m_sel = sa > 0
    m_ref = ra > 0
    if not m_sel.any() or not m_ref.any():
        return src_bgra.copy()

    corrected = np.empty_like(src_bgr)
    for c in range(3):
        lut = _weighted_lut(src_bgr[:, :, c][m_sel],
                            ref_bgr[:, :, c][m_ref],
                            sa[m_sel], ra[m_ref])
        corrected[:, :, c] = cv2.LUT(src_bgr[:, :, c], lut)

    out = src_bgra.copy()
    if strength >= 0.999:
        out[m_sel, :3] = corrected[m_sel]
    else:
        mix = (src_bgr.astype(np.float32) * (1 - strength)
               + corrected.astype(np.float32) * strength)
        out[m_sel, :3] = np.clip(mix[m_sel], 0, 255).astype(np.uint8)
    return out


def color_correct(src_bgra: np.ndarray, ref_bgra: np.ndarray,
                  method: str, strength: float) -> np.ndarray:
    """色调统一入口：method = 'reinhard' 或 'hist'"""
    if strength <= 0:
        return src_bgra.copy()
    if method == "hist":
        return hist_match_transfer(src_bgra, ref_bgra, strength)
    return reinhard_transfer(src_bgra, ref_bgra, strength)


# ===================== 批量矫色线程 =====================

class ColorWorker(QThread):
    progress = Signal(int, int, str)
    all_done = Signal(int, int)

    def __init__(self, tasks: List[tuple], ref_bgra: np.ndarray,
                 method: str, strength: float):
        # tasks: [(src_path, dst_path), ...]
        super().__init__()
        self.tasks = tasks
        self.ref_bgra = ref_bgra
        self.method = method
        self.strength = strength

    def run(self):
        ok = fail = 0
        total = len(self.tasks)
        for i, (src_path, dst_path) in enumerate(self.tasks, 1):
            try:
                img = imread_unicode(src_path)
                if img is None:
                    raise IOError(f"读取失败 {src_path}")
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                out = color_correct(img, self.ref_bgra,
                                    self.method, self.strength)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if not imwrite_unicode(dst_path, out):
                    raise IOError(f"写入失败 {dst_path}")
                ok += 1
            except Exception as e:  # 单张失败不影响整批
                fail += 1
                self.progress.emit(i, total, f"失败：{e}")
                continue
            self.progress.emit(i, total, f"处理中 {i}/{total}")
        self.all_done.emit(ok, fail)


# ===================== 缩略图单元格 =====================

class ColorCell(QFrame):
    """色调统一缩略图：黄框=当前，绿框=已矫色"""

    clicked = Signal(str)

    def __init__(self, key: str, thumb_path: str,
                 thumb_w: int = 110, thumb_h: int = 80):
        super().__init__()
        self.key = key
        self._current = False
        self._done = False

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
        from PySide6.QtGui import QPainter
        from gui_frame_matting import _checker_tile
        p = QPainter(canvas)
        p.drawTiledPixmap(canvas.rect(), _checker_tile(10))
        if not src.isNull():
            scaled = src.scaled(self._thumb_w, self._thumb_h,
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap((self._thumb_w - scaled.width()) // 2,
                         (self._thumb_h - scaled.height()) // 2, scaled)
        p.end()
        self.img_label.setPixmap(canvas)

    def set_state(self, current: bool, done: bool, number: int):
        self._current, self._done = current, done
        mark = "✓ " if done else ""
        self.caption.setText(f"{mark}#{number}")
        self._update_style()

    def _update_style(self):
        if self._current:
            border, bg, fg = "#f1c40f", "#1a3a5c", "#f1c40f"
        elif self._done:
            border, bg, fg = "#2ecc71", "#14301f", "#2ecc71"
        else:
            border, bg, fg = "#444444", "#1e1e1e", "#888888"
        self.setStyleSheet(
            f"ColorCell{{background:{bg};border:3px solid {border};"
            f"border-radius:4px;}}"
            f"QLabel{{background:transparent;color:{fg};font-size:12px;}}")

    def mousePressEvent(self, event):
        self.clicked.emit(self.key)


# ===================== 自适应图片显示 =====================

class FitImageLabel(QLabel):
    """按控件大小等比缩放显示 QPixmap，尺寸变化自动重排"""

    def __init__(self, checker: bool = False):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 120)
        self.setStyleSheet("background:#222;border:1px solid #444;")
        self._pix: Optional[QPixmap] = None
        self._checker = checker

    def set_image(self, pix: Optional[QPixmap]):
        self._pix = pix
        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _render(self):
        if self._pix is None or self._pix.isNull():
            self.clear()
            return
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation)
        self.setPixmap(compose_checkerboard(scaled)
                       if self._checker else scaled)


def bgra_to_pixmap(bgra: np.ndarray, checker: bool = True) -> QPixmap:
    """BGRA numpy → QPixmap。checker=False 时保留透明通道（由显示控件铺棋盘格）"""
    from PySide6.QtGui import QImage, QPainter
    if bgra.shape[2] == 3:
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w,
                      QImage.Format_RGB888).copy()
    else:
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, 4 * w,
                      QImage.Format_RGBA8888).copy()
        if not checker:
            return QPixmap.fromImage(qimg)
        # 先铺棋盘格，透明区可见
        pix = QPixmap.fromImage(qimg)
        canvas = QPixmap(pix.size())
        from gui_frame_matting import _checker_tile
        pp = QPainter(canvas)
        pp.drawTiledPixmap(canvas.rect(), _checker_tile(12))
        pp.drawPixmap(0, 0, pix)
        pp.end()
        return canvas
    return QPixmap.fromImage(qimg)


def bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    from PySide6.QtGui import QImage
    qimg = QImage(rgb.data, w, h, 3 * w,
                  QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ===================== 色调统一页面 =====================

class FrameColorPage(QWidget):

    def __init__(self, shared_state: dict):
        super().__init__()
        self.shared_state = shared_state
        self.setStyleSheet("QWidget{background:#2b2b2b;color:#ecf0f1;}")

        # 数据
        self.frames_dir = ""
        self.matted_dir = ""
        self.refined_dir = ""
        self.colored_dir = ""
        self.items: List[dict] = []
        self.cells: List[ColorCell] = []
        self.cur_key: Optional[str] = None
        self.board_key: Optional[str] = None
        self.board_mode = "frame"          # "frame"=取自帧 / "custom"=用户上传
        self.board_custom_path = ""        # 上传基板在 colored 目录内的副本
        self._batch_id = ""
        self._batch_mark: Optional[float] = None
        self.worker: Optional[ColorWorker] = None

        # 预览缓存（当前图片）
        self._src_img: Optional[np.ndarray] = None     # 待矫色 BGRA
        self._board_img: Optional[np.ndarray] = None   # 基板主体 BGRA
        self._preview_img: Optional[np.ndarray] = None

        # resize 防抖
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._build_grid)

        # 实时预览防抖
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ===== 左侧：左右两个显示区 =====
        left_box = QGroupBox("矫色预览（左：抠好的图片 / 右：颜色参考基板原图）")
        lv = QVBoxLayout(left_box)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter{background:#2b2b2b;}")

        top_group = QGroupBox("抠好的图片（矫色对象，棋盘格=透明）")
        tv = QVBoxLayout(top_group)
        self.view_label = FitImageLabel(checker=True)
        tv.addWidget(self.view_label)
        splitter.addWidget(top_group)

        bottom_group = QGroupBox("颜色参考图（基板原图）")
        bv = QVBoxLayout(bottom_group)
        self.board_label = FitImageLabel(checker=False)
        bv.addWidget(self.board_label)
        splitter.addWidget(bottom_group)
        splitter.setSizes([400, 400])
        lv.addWidget(splitter)

        root.addWidget(left_box, 13)

        # ===== 右侧面板 =====
        right_panel = QWidget()
        rlay = QVBoxLayout(right_panel)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(8)

        # 图片导航
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

        # 基板图
        board_box = QGroupBox("基板图（颜色基准）")
        bl = QVBoxLayout(board_box)
        bl.setSpacing(4)
        self.board_name = QLabel("基板：（未选择）")
        self.board_name.setWordWrap(True)
        self.board_name.setStyleSheet("color:#f1c40f;font-size:12px;")
        bl.addWidget(self.board_name)
        btn_upload = QPushButton("📁 上传基板图…")
        btn_upload.setToolTip("选择外部图片作为全部图片的颜色参考基准\n"
                              "（副本保存在输出目录，随批次保留）")
        btn_upload.clicked.connect(self._upload_board_dialog)
        bl.addWidget(btn_upload)
        brow = QHBoxLayout()
        btn_set_board = QPushButton("当前图设为基板")
        btn_set_board.clicked.connect(self._set_board_current)
        btn_first_board = QPushButton("恢复第1张")
        btn_first_board.clicked.connect(self._reset_board_first)
        brow.addWidget(btn_set_board)
        brow.addWidget(btn_first_board)
        bl.addLayout(brow)
        tip = QLabel("可上传任意外部图片作为颜色基准，或用下方按钮选某一帧；\n"
                     "统计只取主体像素（上传的无透明通道图片整幅参与），\n"
                     "切换基板后所有图片以新基板为准")
        tip.setStyleSheet("color:#888;font-size:11px;")
        bl.addWidget(tip)
        rlay.addWidget(board_box, 0)

        # 矫色设置
        set_box = QGroupBox("矫色设置")
        sl = QVBoxLayout(set_box)
        sl.setSpacing(4)
        method_row = QHBoxLayout()
        self.method_group = QButtonGroup(self)
        self.rb_reinhard = QRadioButton("Reinhard 迁移")
        self.rb_hist = QRadioButton("直方图匹配")
        self.rb_hist.setChecked(True)
        self.rb_reinhard.setToolTip("Lab 空间对齐均值/标准差，色调自然")
        self.rb_hist.setToolTip("逐通道匹配直方图分布，对齐更强")
        for rb in (self.rb_hist, self.rb_reinhard):
            self.method_group.addButton(rb)
            method_row.addWidget(rb)
            rb.toggled.connect(self._schedule_preview)
        sl.addLayout(method_row)

        str_row = QHBoxLayout()
        str_row.addWidget(QLabel("强度:"))
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_label = QLabel("100%")
        self.strength_label.setMinimumWidth(34)
        self.strength_slider.valueChanged.connect(
            lambda v: self.strength_label.setText(f"{v}%"))
        self.strength_slider.valueChanged.connect(self._schedule_preview)
        str_row.addWidget(self.strength_slider, 1)
        str_row.addWidget(self.strength_label)
        sl.addLayout(str_row)

        view_row = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        self.rb_view_result = QRadioButton("矫色结果")
        self.rb_view_before = QRadioButton("矫色前")
        self.rb_view_result.setChecked(True)
        for rb in (self.rb_view_result, self.rb_view_before):
            self.view_group.addButton(rb)
            view_row.addWidget(rb)
            rb.toggled.connect(self._render_current)
        sl.addLayout(view_row)
        rlay.addWidget(set_box, 0)

        # 操作
        op_box = QGroupBox("操作")
        ol = QVBoxLayout(op_box)
        ol.setSpacing(4)
        save_row = QHBoxLayout()
        self.btn_save = QPushButton("💾 保存当前")
        self.btn_save.clicked.connect(self._save_current)
        self.btn_save_all = QPushButton("📦 全部矫色保存")
        self.btn_save_all.clicked.connect(self._save_all)
        save_row.addWidget(self.btn_save)
        save_row.addWidget(self.btn_save_all)
        ol.addLayout(save_row)
        clear_row = QHBoxLayout()
        self.btn_clear = QPushButton("清除当前矫色")
        self.btn_clear.clicked.connect(self._clear_current)
        self.btn_open = QPushButton("打开输出目录")
        self.btn_open.clicked.connect(self._open_out_dir)
        clear_row.addWidget(self.btn_clear)
        clear_row.addWidget(self.btn_open)
        ol.addLayout(clear_row)
        self.chk_skip = QCheckBox("批量时跳过已矫色图片")
        ol.addWidget(self.chk_skip)
        rlay.addWidget(op_box, 0)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#7f8c8d;font-size:11px;")
        rlay.addWidget(self.status, 0)
        rlay.addStretch(1)

        # 下半：全部图片网格
        self.grid_group = QGroupBox("全部图片（黄框=当前，绿框=已矫色）")
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
        wrap_scroll.setStyleSheet(
            "QScrollArea{background:#2b2b2b;border:none;}")
        wrap_scroll.setWidget(wrap)
        root.addWidget(wrap_scroll, 8)

    # ---------- 批次/加载 ----------

    def showEvent(self, event):
        super().showEvent(event)
        self._reload_if_changed()

    def _reload_if_changed(self):
        matted_dir = self.shared_state.get("matted_dir", "")
        frames_dir = self.shared_state.get("frames_dir", "")
        refined_dir = self.shared_state.get("refined_dir", "")
        if refined_dir and not os.path.isdir(refined_dir):
            refined_dir = ""
        if not matted_dir or not os.path.isdir(matted_dir):
            self._show_empty("尚未找到批量抠图结果\n\n请先在「批量抠图」页面处理选中帧")
            return
        files = sorted(f for f in os.listdir(matted_dir)
                       if f.lower().endswith(".png"))
        batch_id = self._batch_identity(frames_dir, matted_dir, refined_dir)
        if batch_id == self._batch_id:
            return
        self._batch_mark = self._batch_mtime(matted_dir, refined_dir)

        new_colored = os.path.join(frames_dir, "colored") if frames_dir else \
            os.path.join(os.path.dirname(matted_dir), "colored")
        batch_changed = bool(self._batch_id) and self._batch_id != batch_id
        if batch_changed:
            self._discard_stale(self.colored_dir, new_colored)

        self._batch_id = batch_id
        self.frames_dir = frames_dir
        self.matted_dir = matted_dir
        self.refined_dir = refined_dir
        self.colored_dir = new_colored
        os.makedirs(self.colored_dir, exist_ok=True)

        self.items = []
        for i, fn in enumerate(files):
            stem = os.path.splitext(fn)[0]
            matted_path = os.path.join(matted_dir, fn)
            refined_path = os.path.join(refined_dir, fn) if refined_dir else ""
            colored_path = os.path.join(self.colored_dir, fn)
            orig = self._find_original(stem)
            self.items.append({
                "key": fn, "number": i + 1,
                "matted": matted_path,
                "refined": refined_path,
                "colored": colored_path,
                "orig": orig,
            })

        # 恢复上次基板/方法/强度
        saved = self._read_info()
        if batch_changed:
            self.board_mode = "frame"
            self.board_custom_path = ""
        saved_board = saved.get("board_key", "")
        saved_mode = saved.get("board_mode", "frame")
        saved_custom = saved.get("board_file", "")
        if saved_mode == "custom" and saved_custom:
            cand = os.path.join(self.colored_dir, saved_custom)
            if os.path.isfile(cand) and self._load_bgra(cand) is not None:
                self.board_mode = "custom"
                self.board_custom_path = cand
            else:
                self.board_mode = "frame"
                self.board_custom_path = ""
        else:
            self.board_mode = "frame"
            self.board_custom_path = ""
        # 缺省（旧记录或首次进入）使用直方图匹配
        if saved.get("method") == "reinhard":
            self.rb_reinhard.setChecked(True)
        else:
            self.rb_hist.setChecked(True)
        st = int(saved.get("strength", 100))
        self.strength_slider.setValue(max(0, min(100, st)))

        keys = {it["key"] for it in self.items}
        if self.board_key not in keys:
            self.board_key = saved_board if saved_board in keys else (
                self.items[0]["key"] if self.items else None)
        if self.cur_key not in keys:
            self.cur_key = self.items[0]["key"] if self.items else None

        self._build_grid()
        self._open_current()
        n_done = sum(1 for it in self.items
                     if self._is_current(it["colored"]))
        self.status_text(f"已加载 {len(self.items)} 张，已矫色 {n_done} 张")

    @staticmethod
    def _sig(directory: str) -> str:
        if not directory:
            return ""
        info = os.path.join(directory, "matted_info.json")
        if not os.path.isfile(info):
            info = os.path.join(directory, "refined_info.json")
        if os.path.isfile(info):
            try:
                return f"info:{os.path.getmtime(info):.3f}"
            except OSError:
                pass
        try:
            return "files:" + "|".join(sorted(
                f for f in os.listdir(directory)
                if f.lower().endswith(".png")))
        except OSError:
            return ""

    def _batch_identity(self, frames_dir, matted_dir, refined_dir) -> str:
        return "|".join((
            os.path.abspath(frames_dir) if frames_dir else "",
            os.path.abspath(matted_dir),
            self._sig(matted_dir),
            self._sig(refined_dir),
        ))

    @staticmethod
    def _batch_mtime(matted_dir, refined_dir) -> Optional[float]:
        marks = []
        for d, name in ((matted_dir, "matted_info.json"),
                        (refined_dir, "refined_info.json")):
            if d:
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    try:
                        marks.append(os.path.getmtime(p))
                    except OSError:
                        pass
        return max(marks) if marks else None

    def _is_current(self, path: str) -> bool:
        """矫色结果是否属于当前批次（防旧文件残留顶替）"""
        if not path or not os.path.isfile(path):
            return False
        if self._batch_mark is None:
            return True
        try:
            return os.path.getmtime(path) >= self._batch_mark - 1e-6
        except OSError:
            return False

    def _discard_stale(self, old_dir: str, new_dir: str):
        if not old_dir or not new_dir:
            return
        if os.path.basename(os.path.normpath(old_dir)).lower() != "colored":
            return
        if os.path.abspath(old_dir) != os.path.abspath(new_dir):
            return
        shutil.rmtree(old_dir, ignore_errors=True)
        if os.path.abspath(self.shared_state.get("colored_dir", "")) \
                == os.path.abspath(old_dir):
            self.shared_state["colored_dir"] = ""

    def _find_original(self, stem: str) -> Optional[str]:
        if not self.frames_dir or not os.path.isdir(self.frames_dir):
            return None
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            p = os.path.join(self.frames_dir, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def _show_empty(self, text: str):
        self.items, self.cells = [], []
        self.cur_key = self.board_key = None
        self.board_mode = "frame"
        self.board_custom_path = ""
        self._src_img = self._board_img = self._preview_img = None
        self._batch_id = ""
        self.view_label.set_image(None)
        self.board_label.set_image(None)
        self.board_name.setText("基板：（未选择）")
        self.pos_label.setText("- / -")
        while self.grid_layout.count():
            it = self.grid_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.status_text(text)

    # ---------- 数据读取 ----------

    def _cur_item(self) -> Optional[dict]:
        return next((it for it in self.items
                     if it["key"] == self.cur_key), None)

    def _item(self, key: str) -> Optional[dict]:
        return next((it for it in self.items if it["key"] == key), None)

    def _source_path(self, item: dict) -> str:
        """精抠优先，回退批量抠图"""
        if item.get("refined") and os.path.isfile(item["refined"]):
            return item["refined"]
        return item["matted"]

    @staticmethod
    def _load_bgra(path: str) -> Optional[np.ndarray]:
        img = imread_unicode(path)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    def _method(self) -> str:
        return "hist" if self.rb_hist.isChecked() else "reinhard"

    # ---------- 显示 / 预览 ----------

    def _open_current(self):
        it = self._cur_item()
        self._src_img = self._preview_img = None
        self._board_img = None
        if it is None:
            self.view_label.set_image(None)
            self.board_label.set_image(None)
            self.board_name.setText("基板：（未选择）")
            self.pos_label.setText("- / -")
            self._update_ops()
            self._refresh_cells()
            return
        self._src_img = self._load_bgra(self._source_path(it))
        self._apply_board()
        idx = self.items.index(it) + 1
        self.pos_label.setText(f"{idx} / {len(self.items)}  {it['key']}")
        self._update_preview()
        self._refresh_cells()
        self._update_ops()

    def _apply_board(self) -> bool:
        """按 board_mode 载入基板统计图（BGRA）并刷新下方显示。
        custom=用户上传图片（无透明通道则整幅作为主体参与统计）；
        frame=某一帧抠好后的主体，下方显示该帧原图。"""
        if self.board_mode == "custom" and self.board_custom_path:
            img = self._load_bgra(self.board_custom_path)
            if img is not None:
                self._board_img = img
                self.board_label.set_image(bgra_to_pixmap(img))
                self.board_name.setText(
                    "基板（上传）：" + os.path.basename(self.board_custom_path))
                return True
            # 副本丢失则回退帧模式
            self.board_mode = "frame"
            self.board_custom_path = ""
        board_item = self._item(self.board_key) if self.board_key else None
        if board_item is not None:
            self._board_img = self._load_bgra(
                self._source_path(board_item))
            board_disp_path = board_item["orig"] or \
                self._source_path(board_item)
            disp = imread_unicode(board_disp_path)
            self.board_label.set_image(
                bgr_to_pixmap(disp[:, :, :3]) if disp is not None else None)
            self.board_name.setText(
                f"基板：{self.board_key}（第 {board_item['number']} 张）")
            return self._board_img is not None
        self._board_img = None
        self.board_label.set_image(None)
        self.board_name.setText("基板：（未选择）")
        return False

    def _schedule_preview(self, *args):
        if self._src_img is not None and self._board_img is not None:
            self._preview_timer.start()

    def _update_preview(self):
        if self._src_img is None or self._board_img is None:
            self._render_current()
            self._update_ops()
            return
        self._preview_img = color_correct(
            self._src_img, self._board_img,
            self._method(), self.strength_slider.value() / 100.0)
        self._render_current()
        self._update_ops()

    def _render_current(self, *args):
        """按当前视图单选渲染上图"""
        if self.rb_view_before.isChecked() and self._src_img is not None:
            img = self._src_img
        else:
            img = self._preview_img if self._preview_img is not None \
                else self._src_img
        self.view_label.set_image(
            bgra_to_pixmap(img) if img is not None else None)

    def _update_ops(self):
        has = self._src_img is not None and self._board_img is not None
        busy = self.worker is not None and self.worker.isRunning()
        self.btn_save.setEnabled(has and not busy)
        self.btn_save_all.setEnabled(bool(self.items) and not busy)
        self.btn_clear.setEnabled(has and not busy)
        it = self._cur_item()
        done = bool(it and self._is_current(it["colored"]))
        if done:
            self.state_label.setText("✓ 已矫色保存")
            self.state_label.setStyleSheet("color:#2ecc71;")
        else:
            self.state_label.setText("未保存")
            self.state_label.setStyleSheet("color:#888;")

    # ---------- 交互 ----------

    def _switch_to(self, key: str):
        if key == self.cur_key:
            return
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

    def _set_board_current(self):
        if not self.cur_key:
            return
        self.board_mode = "frame"
        self.board_custom_path = ""
        self.board_key = self.cur_key
        self._write_info()
        self._apply_board()
        self._schedule_preview()

    def _reset_board_first(self):
        if not self.items:
            return
        self.board_mode = "frame"
        self.board_custom_path = ""
        self.board_key = self.items[0]["key"]
        self._write_info()
        self._apply_board()
        self._schedule_preview()

    # ---------- 用户上传基板 ----------

    def _upload_board_dialog(self):
        if not self.items or not self.colored_dir:
            QMessageBox.information(
                self, "上传基板图",
                "请先完成批量抠图并进入本页面后再上传基板图。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择基板图（颜色参考）", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if path:
            self._load_custom_board(path)

    def _load_custom_board(self, src_path: str) -> bool:
        """把外部图片复制进输出目录并设为基板（随批次保留/清理）"""
        img = self._load_bgra(src_path)
        if img is None:
            QMessageBox.warning(self, "上传基板图", f"无法读取图片：\n{src_path}")
            return False
        os.makedirs(self.colored_dir, exist_ok=True)
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            ext = ".png"
        dst_name = "_board_ref" + ext
        dst_path = os.path.join(self.colored_dir, dst_name)
        # 删掉旧的不同扩展名副本，避免恢复时读到两个
        for old in os.listdir(self.colored_dir):
            if os.path.splitext(old)[0] == "_board_ref" and old != dst_name:
                try:
                    os.remove(os.path.join(self.colored_dir, old))
                except OSError:
                    pass
        try:
            shutil.copyfile(src_path, dst_path)
        except OSError:
            # 复制失败则直接引用原路径（不落副本，重启后需重新上传）
            dst_path = src_path
            dst_name = os.path.basename(src_path)
        self.board_mode = "custom"
        self.board_custom_path = dst_path
        self._write_info()
        self._apply_board()
        self._schedule_preview()
        self.status_text(f"已使用上传基板图：{os.path.basename(src_path)}")
        return True

    # ---------- 保存 ----------

    def _info_path(self) -> str:
        return os.path.join(self.colored_dir, "colored_info.json")

    def _read_info(self) -> dict:
        try:
            with open(self._info_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_info(self):
        os.makedirs(self.colored_dir, exist_ok=True)
        data = {
            "board_mode": self.board_mode,
            "board_key": self.board_key or "",
            "board_file": (os.path.basename(self.board_custom_path)
                           if self.board_mode == "custom"
                           and self.board_custom_path
                           and os.path.dirname(
                               os.path.abspath(self.board_custom_path))
                           == os.path.abspath(self.colored_dir) else ""),
            "method": self._method(),
            "strength": self.strength_slider.value(),
            "files": [it["key"] for it in self.items
                      if self._is_current(it["colored"])],
        }
        try:
            with open(self._info_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _save_current(self):
        if self._preview_img is None or self.cur_key is None:
            return
        it = self._cur_item()
        os.makedirs(self.colored_dir, exist_ok=True)
        if imwrite_unicode(it["colored"], self._preview_img):
            self.shared_state["colored_dir"] = self.colored_dir
            self._write_info()
            self._refresh_cells()
            self._update_ops()
            self.status_text(f"已保存：{it['key']}")

    def _save_all(self):
        if not self.items or self._board_img is None:
            return
        targets = []
        for it in self.items:
            if self.chk_skip.isChecked() and self._is_current(it["colored"]):
                continue
            targets.append((self._source_path(it), it["colored"]))
        if not targets:
            self.status_text("所有图片均已矫色，跳过")
            return
        self.worker = ColorWorker(targets, self._board_img,
                                  self._method(),
                                  self.strength_slider.value() / 100.0)
        self.worker.progress.connect(
            lambda d, t, m: self.status_text(f"批量矫色 {d}/{t}  {m}"))
        self.worker.all_done.connect(self._on_all_done)
        self._update_ops()
        self.worker.start()

    def _on_all_done(self, ok: int, fail: int):
        self.shared_state["colored_dir"] = self.colored_dir
        self._write_info()
        self._build_grid()
        self._update_ops()
        self.status_text(f"批量矫色完成：成功 {ok} 张"
                         + (f"，失败 {fail} 张" if fail else ""))

    def _clear_current(self):
        it = self._cur_item()
        if it and os.path.isfile(it["colored"]):
            try:
                os.remove(it["colored"])
            except OSError:
                pass
        self._write_info()
        self._refresh_cells()
        self._update_ops()
        self.status_text("已清除当前图片的矫色结果")

    def _open_out_dir(self):
        if self.colored_dir and os.path.isdir(self.colored_dir):
            os.startfile(os.path.abspath(self.colored_dir))

    def status_text(self, text: str):
        self.status.setText(text)

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
        cols = max(1, avail // (124 + 6))
        for i, item in enumerate(self.items):
            path = item["colored"] if self._is_current(item["colored"]) \
                else self._source_path(item)
            cell = ColorCell(item["key"], path)
            cell.clicked.connect(self._switch_to)
            self.grid_layout.addWidget(cell, i // cols, i % cols)
            self.cells.append(cell)
        self._refresh_cells()

    def _refresh_cells(self):
        for i, item in enumerate(self.items):
            if i < len(self.cells):
                self.cells[i].set_state(
                    current=item["key"] == self.cur_key,
                    done=self._is_current(item["colored"]),
                    number=item["number"])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.items:
            self._resize_timer.start()
