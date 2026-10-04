# -*- coding: utf-8 -*-
"""
模块三：批量抠图 - 页面（可嵌入主窗口）

抠图模式：
  1. 背景色抠图：指定背景色 + 容差 + 羽化，把背景变为透明
  2. rembg 主体抠图：AI 模型自动分离主体

布局与帧数选择页面一致：
  左侧 = 工作区（预览原图 / 抠图结果，透明区域以棋盘格显示）
  右侧上半 = 功能按钮区（帧目录 / 抠图模式 / 执行）
  右侧下半 = 待抠图帧网格（帧数选择页选中帧的实时镜像：
                            蓝框=待处理显示原图，绿框✓=已抠图显示结果）

输出：
  默认保存到 <frames_dir>/matted/，文件名与源帧同名（统一 .png）
  并生成 matted_info.json 供后续模块使用
"""

import os
import json
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QSlider, QFileDialog, QScrollArea, QMessageBox, QGroupBox,
    QSizePolicy, QRadioButton, QButtonGroup, QProgressBar, QColorDialog,
    QFrame, QComboBox, QStackedWidget, QCheckBox,
)


# ===================== 工具函数 =====================

def imread_unicode(path: str) -> Optional[np.ndarray]:
    """支持中文路径的图片读取"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """支持中文路径的图片写入"""
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


_checker_tile_cache: Dict[int, QPixmap] = {}


def _checker_tile(cell: int = 12) -> QPixmap:
    """棋盘格贴片（2x2 格子，可平铺）"""
    if cell in _checker_tile_cache:
        return _checker_tile_cache[cell]
    t = QPixmap(cell * 2, cell * 2)
    t.fill(QColor(70, 70, 70))
    p = QPainter(t)
    p.fillRect(0, 0, cell, cell, QColor(105, 105, 105))
    p.fillRect(cell, cell, cell, cell, QColor(105, 105, 105))
    p.end()
    _checker_tile_cache[cell] = t
    return t


def compose_checkerboard(pix: QPixmap, cell: int = 12) -> QPixmap:
    """把图片合成到棋盘格背景上（用于显示透明区域）"""
    out = QPixmap(pix.size())
    p = QPainter(out)
    p.drawTiledPixmap(out.rect(), _checker_tile(cell))
    p.drawPixmap(0, 0, pix)
    p.end()
    return out


# ===================== 帧引用 =====================

class FrameRef:
    def __init__(self, index: int, filename: str, filepath: str):
        self.index = index
        self.filename = filename
        self.filepath = filepath


# ===================== 色键抠像（chroma key）算法工具 =====================

def border_connected_mask(seed: np.ndarray) -> np.ndarray:
    """
    输入 0/1 掩膜（1=背景候选），只保留与画面四条边连通的候选区域。
    主体内部被前景包围的同色"孤岛"不会被保留 —— 用于保护主体内部颜色。
    """
    num, labels = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.zeros_like(seed, dtype=np.uint8)
    edge_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    edge_labels = [int(x) for x in edge_labels if x != 0]
    if not edge_labels:
        return np.zeros_like(seed, dtype=np.uint8)
    return np.isin(labels, edge_labels).astype(np.uint8)


def despill_edges(bgr: np.ndarray, alpha: np.ndarray,
                  colors: list, strength: float, slack: int = 6,
                  band: int = 2) -> np.ndarray:
    """
    去溢色（抑制背景色在前景边缘的色晕，如绿边/蓝边）。
    经典色键做法：背景色主导分量若高于另外两分量最大值+slack，把超出部分压回。
    作用区：半透明像素 + 其外侧 band px 的不透明边缘（色边常落在不透明侧）。
    白/灰像素 G≈R≈B，天然不受影响；远离边缘的像素不受影响。
    strength: 0.0~2.0（100%=压平到参考线，>100%=压过头，用于顽固色边）
    """
    if strength <= 0:
        return bgr
    out = bgr.copy()
    semi = (alpha > 0) & (alpha < 255)
    if not semi.any():
        return out
    # 半透明边缘向外膨胀 band px，覆盖紧贴边缘的不透明溢色像素
    zone = cv2.dilate(semi.astype(np.uint8),
                      np.ones((band * 2 + 1, band * 2 + 1), np.uint8)).astype(bool)
    zone &= alpha > 0
    if not zone.any():
        return out
    # 半透明像素按透明度加权；不透明边缘像素满权（保证也能压下去）
    w = np.where(alpha < 255,
                 np.maximum(alpha.astype(np.float32) / 255.0, 0.25),
                 1.0) * strength
    for r0, g0, b0 in colors:
        bg = np.array([b0, g0, r0], dtype=np.int16)   # BGR
        dominant = int(np.argmax(bg))
        others = [i for i in range(3) if i != dominant]
        d = out[:, :, dominant].astype(np.int16)
        ref = np.maximum(out[:, :, others[0]].astype(np.int16),
                         out[:, :, others[1]].astype(np.int16))
        over = np.clip(d - ref - slack, 0, 255).astype(np.float32)
        new_d = np.clip(d - (over * w).astype(np.int16), 0, 255)
        out[:, :, dominant] = np.where(
            zone, new_d.astype(np.uint8), out[:, :, dominant])
    return out


def decontaminate_edges(bgr: np.ndarray, alpha: np.ndarray,
                        bg_bgr: np.ndarray, strength: float,
                        a_floor: float = 0.03) -> np.ndarray:
    """
    边缘净化：半透明边缘像素的观察色是 alpha 合成的结果
        C = a*F + (1-a)*B   （F 真实前景色，B 已知背景色）
    已知 alpha 和 B 即可反解真实前景色：F = (C - (1-a)*B) / a。
    与"去溢色"只压背景色主导分量不同，本操作按合成原理把混入的
    背景色整体扣除，对深色、浅色、三分量接近的背景都有效。
    alpha < a_floor 的像素反解数值不稳定，保持原色（视觉贡献可忽略）。
    strength: 0.0~2.0（100%=完全反解，>100% 过度外推用于顽固色边）。
    """
    if strength <= 0:
        return bgr
    a = alpha.astype(np.float32) / 255.0
    semi = a >= a_floor
    if not semi.any():
        return bgr
    src = bgr.astype(np.float32)
    af = np.maximum(a, 1e-3)[..., None]
    est = (src - (1.0 - af) * bg_bgr) / af      # 反解前景色
    est = np.clip(est, 0.0, 255.0)
    out = src + (est - src) * strength          # 按强度向净色靠拢
    out = np.where(semi[..., None], out, src)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def matte_bgcolor_array(bgr: np.ndarray, colors: list, tol: float,
                        feather: int, choke: int, despill_pct: int,
                        protect_inner: bool,
                        edge_clean_pct: int = 0) -> np.ndarray:
    """
    色键抠像流水线（多背景色），输入 BGR 图像，返回 BGRA 结果。
      1. 像素到最近背景色的 RGB 距离 + 软阈值
      2. 连通域保护：只抠除与画面边缘连通的背景，保留主体内部同色
      3. 边缘收缩(choke)：alpha 向内收 N px，吃掉主体边缘残留色边
      4. 羽化：高斯柔化收缩后的硬边
      5. 去溢色(despill)：中和边缘的背景色晕（可 >100% 加强压制）
      6. 边缘净化(clean)：按 alpha 合成公式反解半透明像素的真实前景色
    worker 批量处理与界面实时预览共用此函数。
    """
    need_clean = edge_clean_pct > 0 and bool(colors)

    # 1) 每个像素到所有背景色的距离，取最小值；
    #    边缘净化开启时同时记录每像素命中的背景色（距离最近者）
    dist = None
    best_bg = np.empty(bgr.shape, np.float32) if need_clean else None
    for r, g, b in colors:
        bgv = np.array([b, g, r], np.float32)
        d = np.linalg.norm(bgr.astype(np.float32) - bgv, axis=2)
        if dist is None:
            dist = d
            if need_clean:
                best_bg[:] = bgv
        else:
            m = d < dist
            dist = np.minimum(dist, d)
            if need_clean:
                best_bg[m] = bgv

    # 软阈值：0.75*tol 内全透明，tol 外全不透明，中间线性渐变
    inner = tol * 0.75
    soft = np.clip((dist - inner) / max(tol - inner, 1e-6), 0.0, 1.0)
    alpha = (soft * 255).astype(np.uint8)

    # 2) 连通域保护：只有与画面边缘连通的背景才允许被抠除
    if protect_inner:
        seed = (dist < inner).astype(np.uint8)
        bg_core = border_connected_mask(seed)
        if bg_core.max() > 0:
            dilate_px = max(3, feather + choke + 1)
            k = dilate_px * 2 + 1
            bg_zone = cv2.dilate(bg_core, np.ones((k, k), np.uint8))
            alpha = np.where(bg_zone > 0, alpha, 255).astype(np.uint8)

    # 3) 边缘收缩：erode 让 alpha 边缘向内收，吃掉 1~N px 残留色边
    if choke > 0:
        k = choke * 2 + 1
        alpha = cv2.erode(alpha, np.ones((k, k), np.uint8))

    # 4) 羽化：柔化收缩后的边缘
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    # 5) 去溢色：中和边缘的绿边/蓝边
    bgr = despill_edges(bgr, alpha, colors, despill_pct / 100.0)

    # 6) 边缘净化：反解半透明像素的真实前景色，整体扣除混入的背景色
    if need_clean:
        bgr = decontaminate_edges(bgr, alpha, best_bg,
                                  edge_clean_pct / 100.0)

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = alpha
    return out


# ===================== 后台抠图线程 =====================

class MattingWorker(QThread):
    progress = Signal(int, int, str)   # 已完成数, 总数, 消息
    frame_done = Signal(int, str)      # 帧索引, 输出路径
    all_done = Signal(int, int)        # 成功数, 失败数

    def __init__(self, mode: str, tasks: List[Tuple[int, str, str]],
                 params: dict, parent=None):
        super().__init__(parent)
        self.mode = mode          # "bgcolor" / "rembg"
        self.tasks = tasks        # [(帧索引, 源路径, 输出路径), ...]
        self.params = params
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        ok = fail = 0
        total = len(self.tasks)
        session = None

        if self.mode == "rembg":
            try:
                from rembg import remove, new_session
                model = self.params.get("model", "isnet-general-use")
                self.progress.emit(0, total, f"加载 rembg 模型 {model}（首次使用需下载）...")
                session = new_session(model)
            except Exception as e:
                self.progress.emit(0, total, f"rembg 加载失败: {e}")
                self.all_done.emit(0, total)
                return

        for done, (idx, src, dst) in enumerate(self.tasks):
            if self._stop_flag:
                break
            try:
                if self.mode == "bgcolor":
                    self._matte_bgcolor(src, dst)
                else:
                    self._matte_rembg(src, dst, session)
                ok += 1
                self.frame_done.emit(idx, dst)
            except Exception:
                fail += 1
                traceback.print_exc()
            self.progress.emit(done + 1, total, f"处理中 {done + 1}/{total}")

        self.all_done.emit(ok, fail)

    def _matte_bgcolor(self, src: str, dst: str):
        """色键抠像（实现见模块级 matte_bgcolor_array，本方法负责读写文件）"""
        colors = self.params.get("colors")
        if colors is None:  # 兼容旧版单色参数
            colors = [self.params["color"]]

        img = imread_unicode(src)
        if img is None:
            raise IOError(f"无法读取 {src}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr = img[:, :, :3]

        out = matte_bgcolor_array(
            bgr, colors,
            tol=float(self.params["tolerance"]),
            feather=int(self.params["feather"]),
            choke=int(self.params.get("choke", 1)),
            despill_pct=int(self.params.get("despill", 80)),
            protect_inner=bool(self.params.get("protect_inner", True)),
            edge_clean_pct=int(self.params.get("edge_clean", 0)),
        )

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not imwrite_unicode(dst, out):
            raise IOError(f"写入失败 {dst}")

    def _matte_rembg(self, src: str, dst: str, session):
        """rembg 主体抠图"""
        from PIL import Image
        from rembg import remove
        pil = Image.open(src).convert("RGBA")
        out = remove(pil, session=session)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        out.save(dst)


# ===================== 预览区（支持取色） =====================

class PreviewLabel(QLabel):
    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page

    def mousePressEvent(self, event):
        if self.page._pick_mode and event.button() == Qt.LeftButton:
            self.page._pick_color_at(event.position().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)


# ===================== 帧缩略图（待处理/已处理） =====================

class MattedCell(QFrame):
    clicked = Signal(int)

    def __init__(self, index: int, source_path: str,
                 result_path: Optional[str] = None,
                 thumb_w: int = 110, thumb_h: int = 80):
        super().__init__()
        self.index = index
        self.source_path = source_path
        self._done = result_path is not None
        self._current = False

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
        self._thumb_w = thumb_w
        self._thumb_h = thumb_h
        lay.addWidget(self.img_label, 0, Qt.AlignHCenter)

        self.caption = QLabel(f"#{index + 1}")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setFixedHeight(16)
        lay.addWidget(self.caption, 0, Qt.AlignHCenter)

        self.set_image(result_path if self._done else source_path)
        self._update_style()

    def set_image(self, filepath: str):
        """在固定画布上绘制图片：结果图铺棋盘格，原图用深色底"""
        src = QPixmap(filepath)
        canvas = QPixmap(self._thumb_w, self._thumb_h)
        p = QPainter(canvas)
        if self._done:
            p.drawTiledPixmap(canvas.rect(), _checker_tile(10))
        else:
            p.fillRect(canvas.rect(), QColor(30, 30, 30))
        if not src.isNull():
            scaled = src.scaled(self._thumb_w, self._thumb_h,
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(
                (self._thumb_w - scaled.width()) // 2,
                (self._thumb_h - scaled.height()) // 2, scaled)
        p.end()
        self.img_label.setPixmap(canvas)

    def set_done(self, result_path: Optional[str]):
        """切换为已处理（传 None 回到待处理原图）"""
        self._done = result_path is not None
        self.set_image(result_path if self._done else self.source_path)
        self._update_style()

    def is_done(self) -> bool:
        return self._done

    def set_current(self, cur: bool):
        if cur != self._current:
            self._current = cur
            self._update_style()

    def is_current(self) -> bool:
        return self._current

    def _update_style(self):
        if self._current:
            # 当前工作区显示的帧：黄色边框（优先级最高，无论是否已抠图）
            self.setStyleSheet(
                "QFrame{border: 3px solid #f1c40f; border-radius: 4px; "
                "background: #4a3d12;}")
            mark = "● " if self._done else ""
            self.caption.setText(f"{mark}#{self.index + 1}")
            self.caption.setStyleSheet(
                "font-size: 10px; color: #f1c40f; font-weight: bold;")
        elif self._done:
            # 已抠图：绿色边框 + ✓
            self.setStyleSheet(
                "QFrame{border: 3px solid #27ae60; border-radius: 4px; "
                "background: #1e3a26;}")
            self.caption.setText(f"✓ #{self.index + 1}")
            self.caption.setStyleSheet(
                "font-size: 10px; color: #2ecc71; font-weight: bold;")
        else:
            # 待处理：蓝色边框
            self.setStyleSheet(
                "QFrame{border: 3px solid #2d7ff9; border-radius: 4px; "
                "background: #1a3a5c;}")
            self.caption.setText(f"#{self.index + 1}")
            self.caption.setStyleSheet(
                "font-size: 10px; color: #5dade2; font-weight: bold;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
            event.accept()
        else:
            super().mousePressEvent(event)


# ===================== 页面：批量抠图 =====================

class FrameMattingPage(QWidget):
    REMBG_MODELS = [
        ("isnet-general-use（推荐·通用）", "isnet-general-use"),
        ("u2net（通用）", "u2net"),
        ("u2net_human_seg（人像）", "u2net_human_seg"),
    ]

    def __init__(self, shared_state: dict = None, parent=None):
        super().__init__(parent)
        self.shared_state = shared_state if shared_state is not None else {}
        self.frames: List[FrameRef] = []
        self.cells: List[MattedCell] = []
        self.results: Dict[int, str] = {}   # 帧索引 -> 抠图输出路径
        self.frames_dir: str = ""
        self._frames_sig: str = ""        # 帧目录内容签名（换视频重抽时变化）
        self.output_dir: str = ""
        self.selected: set = set()          # 来自帧数选择的帧索引
        self.cur_index: int = -1
        self.cur_image: Optional[QImage] = None
        self._pick_mode: bool = False
        self._bg_colors: List[Tuple[int, int, int]] = [(0, 255, 0)]
        self.worker: Optional[MattingWorker] = None

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._build_grid)

        # 参数实时预览防抖（滑块拖动停止后再算，避免卡顿）
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._live_preview)

        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        d = self.shared_state.get("frames_dir", "")
        if d and os.path.isdir(d):
            sig = self._frames_signature(d)
            if d != self.frames_dir or sig != self._frames_sig:
                # 换了帧目录，或同一目录被重新抽帧（签名变化）→ 完整重载，
                # 清掉旧视频的帧列表/抠图结果缓存
                self.dir_edit.setText(d)
                self._load_frames(d)
            elif self.frames:
                # 同一批帧：用户可能刚在帧数选择页改过选择，每次进入都刷新
                self._refresh_selection()

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

    def _read_selected(self) -> set:
        """读取选中帧索引：优先内存共享（实时），其次 selected_frames.json"""
        shared_dir = self.shared_state.get("frames_dir", "")
        indices = self.shared_state.get("selected_indices")
        if indices is not None and os.path.normpath(
                shared_dir) == os.path.normpath(self.frames_dir):
            return set(indices)
        sel_path = os.path.join(self.frames_dir, "selected_frames.json")
        if os.path.isfile(sel_path):
            try:
                with open(sel_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return set(data.get("selected_indices", []))
            except Exception:
                pass
        return set()

    def _ordered_selected(self) -> List[int]:
        """当前选中帧的有序索引列表（即下方网格的镜像顺序）"""
        return sorted(i for i in self.selected if 0 <= i < len(self.frames))

    def _refresh_selection(self):
        n = len(self.frames)
        self.selected = {i for i in self._read_selected() if 0 <= i < n}
        self._update_sel_label()
        # 网格是选中帧的实时镜像，选择变化时整体重建
        self._build_grid()
        # 校正当前预览帧：不在选中集合则跳到第一张选中帧
        ordered = self._ordered_selected()
        if self.cur_index not in self.selected:
            self.cur_index = ordered[0] if ordered else -1
        self._show_current()

    # ---------- UI 构建 ----------

    def _build_ui(self):
        self.setStyleSheet(
            "QGroupBox{border:none;margin-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        self.h_layout = QHBoxLayout()
        self.h_layout.setSpacing(8)
        root.addLayout(self.h_layout, 1)

        # ===== 左侧：工作区 =====
        work_box = QGroupBox("工作区")
        wl = QVBoxLayout(work_box)

        self.preview_label = PreviewLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setText("未加载帧")
        self.preview_label.setStyleSheet(
            "QLabel{background:#2b2b2b;color:#bdc3c7;border:1px solid #444;}")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        wl.addWidget(self.preview_label, 1)

        self.h_layout.addWidget(work_box, 13)

        # ===== 右侧面板 =====
        right_panel = QWidget()
        rlay = QVBoxLayout(right_panel)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(8)

        # --- 帧目录 ---
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

        # --- 抠图模式 ---
        mode_box = QGroupBox("抠图模式")
        ml = QVBoxLayout(mode_box)

        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.rb_mode_bg = QRadioButton("背景色抠图")
        self.rb_mode_rembg = QRadioButton("rembg 主体抠图")
        self.rb_mode_bg.setChecked(True)
        self.mode_group.addButton(self.rb_mode_bg)
        self.mode_group.addButton(self.rb_mode_rembg)
        self.rb_mode_bg.toggled.connect(self._switch_mode)
        mode_row.addWidget(self.rb_mode_bg)
        mode_row.addWidget(self.rb_mode_rembg)
        mode_row.addStretch()
        ml.addLayout(mode_row)

        # 模式参数（堆叠切换）
        self.params_stack = QStackedWidget()
        self.params_stack.setMaximumHeight(294)

        # 背景色抠图参数页
        bg_page = QWidget()
        bgl = QVBoxLayout(bg_page)
        bgl.setContentsMargins(0, 0, 0, 0)
        bgl.setSpacing(4)

        tip_row = QHBoxLayout()
        tip_row.addWidget(QLabel("背景色（可添加多个，命中任一颜色即去除）:"))
        tip_row.addStretch()
        bgl.addLayout(tip_row)

        # 色块列表（自动换行）+ 添加按钮
        self.color_grid = QGridLayout()
        self.color_grid.setSpacing(4)
        bgl.addLayout(self.color_grid)

        add_row = QHBoxLayout()
        self.btn_add_color = QPushButton("＋ 选色添加")
        self.btn_add_color.clicked.connect(self._choose_color)
        add_row.addWidget(self.btn_add_color)
        self.btn_pick = QPushButton("🎨 从图取色添加")
        self.btn_pick.setCheckable(True)
        self.btn_pick.toggled.connect(self._toggle_pick_mode)
        add_row.addWidget(self.btn_pick)
        add_row.addStretch()
        bgl.addLayout(add_row)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("容差:"))
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(0, 150)
        self.tol_slider.setValue(40)
        self.tol_label = QLabel("40")
        self.tol_label.setMinimumWidth(30)
        self.tol_slider.valueChanged.connect(
            lambda v: self.tol_label.setText(str(v)))
        tol_row.addWidget(self.tol_slider, 1)
        tol_row.addWidget(self.tol_label)
        bgl.addLayout(tol_row)

        feather_row = QHBoxLayout()
        feather_row.addWidget(QLabel("羽化:"))
        self.feather_slider = QSlider(Qt.Horizontal)
        self.feather_slider.setRange(0, 10)
        self.feather_slider.setValue(2)
        self.feather_label = QLabel("2")
        self.feather_label.setMinimumWidth(30)
        self.feather_slider.valueChanged.connect(
            lambda v: self.feather_label.setText(str(v)))
        feather_row.addWidget(self.feather_slider, 1)
        feather_row.addWidget(self.feather_label)
        bgl.addLayout(feather_row)

        choke_row = QHBoxLayout()
        choke_row.addWidget(QLabel("边缘收缩:"))
        self.choke_slider = QSlider(Qt.Horizontal)
        self.choke_slider.setRange(0, 4)
        self.choke_slider.setValue(0)
        self.choke_label = QLabel("0")
        self.choke_label.setMinimumWidth(30)
        self.choke_slider.valueChanged.connect(
            lambda v: self.choke_label.setText(str(v)))
        choke_row.addWidget(self.choke_slider, 1)
        choke_row.addWidget(self.choke_label)
        bgl.addLayout(choke_row)

        despill_row = QHBoxLayout()
        despill_row.addWidget(QLabel("去溢色边:"))
        self.despill_slider = QSlider(Qt.Horizontal)
        self.despill_slider.setRange(0, 200)
        self.despill_slider.setValue(80)
        self.despill_label = QLabel("80%")
        self.despill_label.setMinimumWidth(34)
        self.despill_slider.valueChanged.connect(
            lambda v: self.despill_label.setText(f"{v}%"))
        despill_row.addWidget(self.despill_slider, 1)
        despill_row.addWidget(self.despill_label)
        bgl.addLayout(despill_row)

        clean_row = QHBoxLayout()
        clean_row.addWidget(QLabel("边缘净化:"))
        self.clean_slider = QSlider(Qt.Horizontal)
        self.clean_slider.setRange(0, 200)
        self.clean_slider.setValue(100)
        self.clean_label = QLabel("100%")
        self.clean_label.setMinimumWidth(34)
        self.clean_slider.valueChanged.connect(
            lambda v: self.clean_label.setText(f"{v}%"))
        clean_row.addWidget(self.clean_slider, 1)
        clean_row.addWidget(self.clean_label)
        bgl.addLayout(clean_row)

        opt_row = QHBoxLayout()
        self.chk_protect_inner = QCheckBox("保护主体内部同色（防误删）")
        self.chk_protect_inner.setChecked(False)
        opt_row.addWidget(self.chk_protect_inner)
        opt_row.addStretch()
        bgl.addLayout(opt_row)

        self.params_stack.addWidget(bg_page)

        # rembg 参数页
        rembg_page = QWidget()
        rl2 = QVBoxLayout(rembg_page)
        rl2.setContentsMargins(0, 0, 0, 0)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        for label, val in self.REMBG_MODELS:
            self.model_combo.addItem(label, val)
        model_row.addWidget(self.model_combo, 1)
        rl2.addLayout(model_row)
        tip = QLabel("首次使用会自动下载模型文件（约 170MB），请耐心等待")
        tip.setStyleSheet("color:#666; font-size:11px;")
        tip.setWordWrap(True)
        rl2.addWidget(tip)
        rl2.addStretch()
        self.params_stack.addWidget(rembg_page)

        ml.addWidget(self.params_stack)
        rlay.addWidget(mode_box, 0)

        # --- 执行 ---
        exec_box = QGroupBox("执行")
        el = QVBoxLayout(exec_box)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出:"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("默认 <帧目录>/matted")
        out_row.addWidget(self.out_edit, 1)
        btn_out = QPushButton("浏览")
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(btn_out)
        el.addLayout(out_row)

        btn_row = QHBoxLayout()
        self.btn_one = QPushButton("处理当前帧")
        self.btn_one.clicked.connect(self._process_current)
        btn_row.addWidget(self.btn_one)

        self.btn_sel = QPushButton("处理选中帧")
        self.btn_sel.clicked.connect(self._process_selected)
        btn_row.addWidget(self.btn_sel)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        btn_row.addWidget(self.btn_cancel)
        el.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setStyleSheet(
            "QProgressBar{border:1px solid #444;border-radius:3px;"
            "background:#1e1e1e;text-align:center;color:#ccc;height:16px;}"
            "QProgressBar::chunk{background:#2d7ff9;}")
        el.addWidget(self.progress)

        btn_open_row = QHBoxLayout()
        btn_open = QPushButton("📁 打开输出目录")
        btn_open.clicked.connect(self._open_out_dir)
        btn_open_row.addWidget(btn_open)
        self.sel_label = QLabel("已选 0 / 0 帧（来自帧数选择）")
        self.sel_label.setStyleSheet("color:#888;")
        btn_open_row.addStretch()
        btn_open_row.addWidget(self.sel_label)
        el.addLayout(btn_open_row)

        rlay.addWidget(exec_box, 0)

        # --- 预览导航（紧邻待抠图帧网格） ---
        nav_box = QGroupBox("预览导航")
        nv = QVBoxLayout(nav_box)
        nv.setSpacing(4)
        nav_row = QHBoxLayout()
        btn_prev = QPushButton("⏮ 上一帧")
        btn_prev.clicked.connect(self._prev_frame)
        nav_row.addWidget(btn_prev)

        btn_next = QPushButton("⏭ 下一帧")
        btn_next.clicked.connect(self._next_frame)
        nav_row.addWidget(btn_next)

        self.pos_label = QLabel("- / -")
        self.pos_label.setStyleSheet("color:#888;")
        nav_row.addWidget(self.pos_label)
        nav_row.addStretch()
        nv.addLayout(nav_row)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("显示:"))
        self.view_group = QButtonGroup(self)
        self.rb_view_orig = QRadioButton("原图")
        self.rb_view_result = QRadioButton("抠图结果")
        self.rb_view_result.setChecked(True)
        self.view_group.addButton(self.rb_view_orig)
        self.view_group.addButton(self.rb_view_result)
        self.rb_view_orig.toggled.connect(self._show_current)
        self.rb_view_result.toggled.connect(self._show_current)
        view_row.addWidget(self.rb_view_orig)
        view_row.addWidget(self.rb_view_result)
        view_row.addStretch()
        nv.addLayout(view_row)

        # 残留背景色检测
        res_row = QHBoxLayout()
        self.chk_residue = QCheckBox("🔍 残留高亮")
        self.chk_residue.setToolTip(
            "把抠图结果中仍然不透明、但颜色接近背景色的像素\n"
            "用品红色标记出来，方便检查有没有抠干净")
        res_row.addWidget(self.chk_residue)
        res_row.addWidget(QLabel("判定:"))
        self.residue_slider = QSlider(Qt.Horizontal)
        self.residue_slider.setRange(10, 120)
        self.residue_slider.setValue(45)
        self.residue_slider.setToolTip("与背景色的颜色距离小于该值即判为可疑残留")
        self.residue_label = QLabel("45")
        self.residue_label.setMinimumWidth(26)
        res_row.addWidget(self.residue_slider, 1)
        res_row.addWidget(self.residue_label)
        nv.addLayout(res_row)
        self.residue_info = QLabel("")
        self.residue_info.setStyleSheet("color:#e056fd; font-size:11px;")
        self.residue_info.setWordWrap(True)
        nv.addWidget(self.residue_info)

        rlay.addWidget(nav_box, 0)

        # --- 图片存放区（抠图结果网格） ---
        self.grid_box = QGroupBox("待抠图帧（与帧数选择实时同步）")
        gl = QVBoxLayout(self.grid_box)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea{background:#2b2b2b;border:none;}")
        self.container = QWidget()
        self.container.setStyleSheet("QWidget{background:#2b2b2b;}")
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(8)
        self.scroll.setWidget(self.container)
        gl.addWidget(self.scroll, 1)
        rlay.addWidget(self.grid_box, 1)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #555;")
        rlay.addWidget(self.status_label)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("QScrollArea{background:#2b2b2b;border:none;}")
        right_scroll.setWidget(right_panel)
        self.h_layout.addWidget(right_scroll, 8)

        # 初始背景色色块
        self._rebuild_color_chips()

        # 参数变化 → 实时预览（80ms 防抖）
        for _s in (self.tol_slider, self.feather_slider,
                   self.choke_slider, self.despill_slider,
                   self.clean_slider):
            _s.valueChanged.connect(self._schedule_preview)
        self.chk_protect_inner.toggled.connect(self._schedule_preview)

        # 残留高亮开关 / 判定阈值 → 刷新结果显示
        self.chk_residue.toggled.connect(self._on_residue_changed)
        self.residue_slider.valueChanged.connect(
            lambda v: self.residue_label.setText(str(v)))
        self.residue_slider.valueChanged.connect(self._on_residue_changed)

    # ---------- 加载 ----------

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择帧目录")
        if d:
            self.dir_edit.setText(d)
            self._load_frames(d)

    def _load_frames(self, directory: str):
        self.frames_dir = directory
        self.frames = []
        self.cells = []
        self.results = {}
        self.selected = set()
        self.cur_index = -1
        self.cur_image = None
        self._frames_sig = self._frames_signature(directory)

        # 输出目录默认值
        out = self.out_edit.text().strip()
        if not out:
            out = os.path.join(directory, "matted")
            self.out_edit.setText(out)
        self.output_dir = out

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        files = sorted(f for f in os.listdir(directory)
                       if f.lower().endswith(exts))
        for i, fn in enumerate(files):
            self.frames.append(FrameRef(i, fn, os.path.join(directory, fn)))

        # 读取帧数选择页的选中列表（shared_state 优先，json 回退）
        # 过滤越界索引：换视频重抽后帧数可能变少，旧选中索引会超出新帧范围
        self.selected = {i for i in self._read_selected()
                         if 0 <= i < len(self.frames)}

        # 扫描已有抠图结果
        if os.path.isdir(self.output_dir):
            for item in self.frames:
                stem = os.path.splitext(item.filename)[0]
                cand = os.path.join(self.output_dir, stem + ".png")
                if os.path.isfile(cand):
                    self.results[item.index] = cand

        self._build_grid()
        self._update_sel_label()
        self._update_grid_title()
        n_sel = len(self.selected)
        self.status_label.setText(
            f"已加载 {len(self.frames)} 帧，选中 {n_sel} 帧，"
            f"已有结果 {len(self.results)} 张")

        ordered = self._ordered_selected()
        self.cur_index = ordered[0] if ordered else -1
        self._show_current()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.out_edit.setText(d)
            self.output_dir = d

    def _open_out_dir(self):
        out = self.out_edit.text().strip() or self.output_dir
        if not out or not os.path.isdir(out):
            QMessageBox.information(self, "提示", "输出目录不存在")
            return
        os.startfile(out)

    # ---------- 工作区预览 ----------

    def _prev_frame(self):
        ordered = self._ordered_selected()
        if not ordered:
            return
        if self.cur_index in ordered:
            pos = ordered.index(self.cur_index)
            self.cur_index = ordered[(pos - 1) % len(ordered)]
        else:
            self.cur_index = ordered[0]
        self._show_current()

    def _next_frame(self):
        ordered = self._ordered_selected()
        if not ordered:
            return
        if self.cur_index in ordered:
            pos = ordered.index(self.cur_index)
            self.cur_index = ordered[(pos + 1) % len(ordered)]
        else:
            self.cur_index = ordered[0]
        self._show_current()

    def _show_current(self):
        ordered = self._ordered_selected()
        if not (0 <= self.cur_index < len(self.frames)):
            self.preview_label.clear()
            self.preview_label.setText(
                "没有待抠图的帧\n\n请先在「帧数选择」页面选中要抠图的帧，\n"
                "选中的帧会自动同步到本页")
            self.pos_label.setText("- / -")
            self.cur_image = None
            self.residue_info.setText("")
            return

        item = self.frames[self.cur_index]
        pos = ordered.index(self.cur_index) + 1 if self.cur_index in ordered else 0
        done_flag = "✓已抠图" if self.cur_index in self.results else "待处理"
        self.pos_label.setText(
            f"{pos} / {len(ordered)}  #{self.cur_index + 1}  {done_flag}")

        # 加载原图（取色用）
        self.cur_image = QImage(item.filepath)
        self._refresh_current_cell()

        # 背景色模式 + 结果视图：用当前参数实时抠图预览（不落盘）
        live = (self.rb_view_result.isChecked()
                and self.rb_mode_bg.isChecked())
        if live:
            self._schedule_preview()
            return

        show_result = (self.rb_view_result.isChecked()
                       and self.cur_index in self.results)
        if show_result:
            self._render_saved_result(self.results[self.cur_index])
            return

        # 原图视图
        self.residue_info.setText("")
        pix = QPixmap(item.filepath)
        if pix.isNull():
            return
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    # ---------- 残留背景色检测 ----------

    def _on_residue_changed(self):
        if self.rb_view_result.isChecked():
            self._schedule_preview()
        else:
            self.residue_info.setText("")

    @staticmethod
    def _ensure_bgra(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
        if arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        return arr

    def _overlay_residue(self, bgra: np.ndarray) -> np.ndarray:
        """找出不透明但颜色接近任一背景色的可疑像素，用品红标记"""
        thr = float(self.residue_slider.value())
        bgr = bgra[:, :, :3].astype(np.int16)
        dmin = None
        for r, g, b in self._bg_colors:
            d = np.linalg.norm(bgr - np.array([b, g, r], np.int16), axis=2)
            dmin = d if dmin is None else np.minimum(dmin, d)
        visible = bgra[:, :, 3] >= 128       # 只检查基本不透明的像素
        mask = (dmin < thr) & visible
        n_res = int(mask.sum())
        # 轻微膨胀，让单个残留点也肉眼可见
        mark = cv2.dilate(mask.astype(np.uint8),
                          np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        out = bgra.copy()
        out[mark] = (255, 0, 255, 255)       # 品红 BGR
        n_vis = max(1, int(visible.sum()))
        self.residue_info.setText(
            f"品红=可疑残留 {n_res} 像素（占不透明区域 {n_res / n_vis * 100:.2f}%）")
        return out

    def _show_bgra(self, bgra: np.ndarray):
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, w * 4,
                      QImage.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.preview_label.setPixmap(compose_checkerboard(pix))

    def _render_saved_result(self, path: str):
        """渲染已落盘的抠图结果（可选残留高亮叠加）"""
        if self.chk_residue.isChecked():
            if not self._bg_colors:
                self.residue_info.setText("请先在背景色列表中添加要检查的颜色")
            else:
                bgra = self._ensure_bgra(imread_unicode(path))
                if bgra is not None:
                    self._show_bgra(self._overlay_residue(bgra))
                    return
        else:
            self.residue_info.setText("")
        pix = QPixmap(path)
        if pix.isNull():
            return
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
        self.preview_label.setPixmap(compose_checkerboard(scaled))

    # ---------- 参数实时预览 ----------

    def _schedule_preview(self):
        """参数/颜色/帧变化后请求一次实时预览（80ms 防抖）"""
        self._preview_timer.start()

    def _live_preview(self):
        """结果视图刷新：背景色模式实时抠图；或对已保存结果做残留高亮"""
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.rb_view_result.isChecked():
            return
        overlay_on = self.chk_residue.isChecked()

        # 1) 背景色模式：用当前参数实时抠当前帧
        bgra = None
        if (self.rb_mode_bg.isChecked() and self._bg_colors
                and 0 <= self.cur_index < len(self.frames)):
            item = self.frames[self.cur_index]
            img = imread_unicode(item.filepath)
            if img is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                bgra = matte_bgcolor_array(
                    img[:, :, :3], list(self._bg_colors),
                    tol=float(self.tol_slider.value()),
                    feather=int(self.feather_slider.value()),
                    choke=int(self.choke_slider.value()),
                    despill_pct=int(self.despill_slider.value()),
                    protect_inner=bool(self.chk_protect_inner.isChecked()),
                    edge_clean_pct=int(self.clean_slider.value()),
                )

        if bgra is not None:
            if overlay_on:
                bgra = self._overlay_residue(bgra)
            else:
                self.residue_info.setText("")
            self._show_bgra(bgra)
            return

        # 2) 静态已保存结果（rembg 结果等）
        src = self.results.get(self.cur_index)
        if src is not None:
            if overlay_on and self._bg_colors:
                bgra = self._ensure_bgra(imread_unicode(src))
                if bgra is not None:
                    self._show_bgra(self._overlay_residue(bgra))
            elif overlay_on:
                self.residue_info.setText("请先在背景色列表中添加要检查的颜色")
            else:
                # 关闭高亮：恢复普通结果显示并清空统计
                self._render_saved_result(src)

    # ---------- 取色 ----------

    def _toggle_pick_mode(self, checked: bool):
        self._pick_mode = checked
        if checked:
            self.preview_label.setCursor(Qt.CrossCursor)
            # 取色时强制显示原图
            self.rb_view_orig.setChecked(True)
            self.status_label.setText("取色模式：单击预览图拾取背景色（Esc 或再点按钮退出）")
        else:
            self.preview_label.setCursor(Qt.ArrowCursor)

    def _pick_color_at(self, pos):
        if self.cur_image is None:
            return
        img = self.cur_image
        pw, ph = img.width(), img.height()
        lw = self.preview_label.width()
        lh = self.preview_label.height()
        scale = min(lw / pw, lh / ph)
        dw, dh = pw * scale, ph * scale
        ox = (lw - dw) / 2
        oy = (lh - dh) / 2
        ix = int((pos.x() - ox) / scale)
        iy = int((pos.y() - oy) / scale)
        if 0 <= ix < pw and 0 <= iy < ph:
            c = img.pixelColor(ix, iy)
            rgb = (c.red(), c.green(), c.blue())
            added = self._add_color(rgb)
            self.btn_pick.setChecked(False)
            tip = f"已拾取背景色 RGB{rgb}"
            self.status_label.setText(
                tip + ("（已添加）" if added else "（该颜色已在列表中）"))

    # ---------- 多背景色管理 ----------

    COLORS_PER_ROW = 7

    def _rebuild_color_chips(self):
        while self.color_grid.count():
            it = self.color_grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        for i, (r, g, b) in enumerate(self._bg_colors):
            chip = QFrame()
            cl = QHBoxLayout(chip)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(1)

            sw = QPushButton()
            sw.setFixedSize(30, 22)
            sw.setToolTip(f"背景色 {i + 1}: RGB({r},{g},{b})\n点击修改")
            sw.setCursor(Qt.PointingHandCursor)
            sw.setStyleSheet(
                f"QPushButton{{background:rgb({r},{g},{b});"
                f"border:1px solid #999;border-radius:3px;}}"
                f"QPushButton:hover{{border:2px solid #fff;}}")
            sw.clicked.connect(lambda checked, idx=i: self._edit_color(idx))
            cl.addWidget(sw)

            btn_del = QPushButton("×")
            btn_del.setFixedSize(16, 22)
            btn_del.setToolTip("删除该颜色")
            btn_del.setCursor(Qt.PointingHandCursor)
            btn_del.setStyleSheet(
                "QPushButton{color:#ccc;background:#555;border:none;"
                "border-radius:2px;font-weight:bold;font-size:12px;}"
                "QPushButton:hover{background:#c0392b;color:white;}")
            btn_del.clicked.connect(lambda checked, idx=i: self._remove_color(idx))
            cl.addWidget(btn_del)

            self.color_grid.addWidget(
                chip, i // self.COLORS_PER_ROW, i % self.COLORS_PER_ROW)

        # 颜色列表变化后刷新实时预览（初始化时无帧，预览槽会自行跳过）
        self._schedule_preview()

    def _add_color(self, rgb: Tuple[int, int, int]) -> bool:
        """添加背景色，重复颜色不重复添加；返回是否为新增"""
        if rgb in self._bg_colors:
            return False
        self._bg_colors.append(rgb)
        self._rebuild_color_chips()
        return True

    def _edit_color(self, idx: int):
        if not (0 <= idx < len(self._bg_colors)):
            return
        c = QColorDialog.getColor(
            QColor(*self._bg_colors[idx]), self, f"修改背景色 {idx + 1}")
        if c.isValid():
            self._bg_colors[idx] = (c.red(), c.green(), c.blue())
            self._rebuild_color_chips()

    def _remove_color(self, idx: int):
        if not (0 <= idx < len(self._bg_colors)):
            return
        if len(self._bg_colors) <= 1:
            QMessageBox.information(self, "提示", "至少保留一个背景色")
            return
        del self._bg_colors[idx]
        self._rebuild_color_chips()

    def _choose_color(self):
        """＋选色添加：打开颜色对话框，选择后加入颜色列表"""
        c = QColorDialog.getColor(QColor(0, 255, 0), self, "选择要去除的背景色")
        if c.isValid():
            rgb = (c.red(), c.green(), c.blue())
            if self._add_color(rgb):
                self.status_label.setText(f"已添加背景色 RGB{rgb}")
            else:
                self.status_label.setText(f"RGB{rgb} 已在列表中")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._pick_mode:
            self.btn_pick.setChecked(False)
        else:
            super().keyPressEvent(event)

    # ---------- 模式切换 ----------

    def _switch_mode(self):
        if self.rb_mode_bg.isChecked():
            self.params_stack.setCurrentIndex(0)
        else:
            self.params_stack.setCurrentIndex(1)
        # 切换模式后刷新工作区（背景色→实时预览，rembg→磁盘结果/原图）
        self._show_current()

    # ---------- 执行 ----------

    def _process_current(self):
        if not self.frames:
            QMessageBox.warning(self, "提示", "未加载帧目录！")
            return
        if not (0 <= self.cur_index < len(self.frames)):
            QMessageBox.information(
                self, "提示",
                "当前没有待抠图的帧，请先在「帧数选择」页面选中帧")
            return
        self._start_worker([self.cur_index])

    def _process_selected(self):
        if not self.frames:
            QMessageBox.warning(self, "提示", "未加载帧目录！")
            return
        targets = sorted(self.selected)
        if not targets:
            ret = QMessageBox.question(
                self, "提示",
                "帧数选择页面没有选中任何帧。\n\n"
                "【是】处理全部帧并清空旧结果\n【否】返回帧数选择页面重新选择",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret == QMessageBox.Yes:
                targets = list(range(len(self.frames)))
            else:
                return
        # 批量处理前清空旧结果，保证结果与当前选中集合严格对应
        self._clear_output()
        self._start_worker(targets)

    def _clear_output(self):
        out = self.out_edit.text().strip() or self.output_dir
        if os.path.isdir(out):
            for fn in os.listdir(out):
                low = fn.lower()
                if low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")) \
                        or fn == "matted_info.json":
                    try:
                        os.remove(os.path.join(out, fn))
                    except OSError:
                        pass
        self.results = {}
        self._build_grid()

    def _collect_bg_params(self) -> dict:
        """统一收集背景色抠图参数（同时用于 worker 与 matted_info.json）"""
        return {
            "colors": [list(c) for c in self._bg_colors],
            "tolerance": self.tol_slider.value(),
            "feather": self.feather_slider.value(),
            "choke": self.choke_slider.value(),
            "despill": self.despill_slider.value(),
            "edge_clean": self.clean_slider.value(),
            "protect_inner": self.chk_protect_inner.isChecked(),
        }

    def _start_worker(self, targets: List[int]):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.rb_mode_bg.isChecked() and not self._bg_colors:
            QMessageBox.warning(self, "提示", "请至少添加一个背景色！")
            return
        out_dir = self.out_edit.text().strip() or self.output_dir
        self.output_dir = out_dir

        # 组装任务
        tasks = []
        for idx in targets:
            item = self.frames[idx]
            stem = os.path.splitext(item.filename)[0]
            dst = os.path.join(out_dir, stem + ".png")
            tasks.append((idx, item.filepath, dst))

        if self.rb_mode_bg.isChecked():
            mode = "bgcolor"
            params = self._collect_bg_params()
        else:
            mode = "rembg"
            params = {"model": self.model_combo.currentData()}

        self.worker = MattingWorker(mode, tasks, params, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.frame_done.connect(self._on_frame_done)
        self.worker.all_done.connect(self._on_all_done)

        self.progress.setRange(0, len(tasks))
        self.progress.setValue(0)
        self.btn_one.setEnabled(False)
        self.btn_sel.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status_label.setText(f"开始处理 {len(tasks)} 帧（{mode}）...")
        self.worker.start()

    def _cancel_processing(self):
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("正在取消...")

    def _on_progress(self, done: int, total: int, msg: str):
        self.progress.setValue(done)
        self.status_label.setText(msg)

    def _on_frame_done(self, idx: int, out_path: str):
        self.results[idx] = out_path
        self._update_cell_done(idx)
        # 当前帧处理完成则自动切到结果显示
        if idx == self.cur_index:
            self.rb_view_result.setChecked(True)
            self._show_current()

    def _on_all_done(self, ok: int, fail: int):
        self.btn_one.setEnabled(True)
        self.btn_sel.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._write_matted_info()
        self.shared_state["matted_dir"] = self.output_dir
        self._update_grid_title()
        self._show_current()
        self.status_label.setText(
            f"处理完成：成功 {ok} 帧，失败 {fail} 帧  |  输出: {self.output_dir}")

    def _write_matted_info(self):
        """写 matted_info.json 供后续模块使用"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.rb_mode_bg.isChecked():
                mode = "bgcolor"
                params = self._collect_bg_params()
            else:
                mode = "rembg"
                params = {"model": self.model_combo.currentData()}

            frames = []
            for idx in sorted(self.selected & set(self.results)):
                item = self.frames[idx]
                frames.append({
                    "index": idx,
                    "source": item.filename,
                    "filename": os.path.basename(self.results[idx]),
                })
            data = {
                "source_dir": os.path.abspath(self.frames_dir),
                "output_dir": os.path.abspath(self.output_dir),
                "mode": mode,
                "params": params,
                "frames": frames,
            }
            info_path = os.path.join(self.output_dir, "matted_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            traceback.print_exc()

    # ---------- 结果网格 ----------

    def _grid_cols(self) -> int:
        cell_w = 110 + 14 + 8
        avail = self.scroll.viewport().width()
        return max(1, avail // cell_w) if avail > 0 else 4

    def _build_grid(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.cells = []

        ordered = self._ordered_selected()
        if not ordered:
            tip = QLabel("暂无待抠图的帧\n\n请在「帧数选择」页面选中帧，\n选中的帧会自动出现在这里（蓝框=待处理）")
            tip.setAlignment(Qt.AlignCenter)
            tip.setStyleSheet("color:#888; font-size:13px;")
            self.grid.addWidget(tip, 0, 0)
            self.grid.setRowStretch(1, 1)
            self.grid.setColumnStretch(0, 1)
            self._update_grid_title()
            return

        # 重置空状态时设置的 stretch
        self.grid.setRowStretch(1, 0)
        self.grid.setColumnStretch(0, 0)
        for pos, idx in enumerate(ordered):
            self._place_cell(pos, idx)
        self.grid.setRowStretch(self.grid.rowCount(), 1)
        self._update_grid_title()

    def _update_cell_done(self, idx: int):
        """某帧抠图完成：把网格中对应格子切换为结果图（不在网格则忽略）"""
        for cell in self.cells:
            if cell.index == idx:
                cell.set_done(self.results.get(idx))
                break
        self._update_grid_title()

    def _place_cell(self, pos: int, idx: int):
        item = self.frames[idx]
        result = self.results.get(idx)   # 有结果显示结果，否则显示原图（待处理）
        cell = MattedCell(idx, item.filepath, result)
        cell.set_current(idx == self.cur_index)
        cell.clicked.connect(self._on_cell_click)
        r, c = divmod(pos, self._grid_cols())
        self.grid.addWidget(cell, r, c)
        self.cells.append(cell)

    def _refresh_current_cell(self):
        """用黄色边框标记工作区当前显示的帧，其余格子恢复蓝/绿"""
        for cell in self.cells:
            cell.set_current(cell.index == self.cur_index)

    def _on_cell_click(self, idx: int):
        self.cur_index = idx
        # 已抠图的看结果，未抠图的看原图
        if idx in self.results:
            self.rb_view_result.setChecked(True)
        else:
            self.rb_view_orig.setChecked(True)
        self._show_current()

    def _update_grid_title(self):
        n_sel = len(self.selected)
        done = len(self.selected & set(self.results))
        self.grid_box.setTitle(
            f"待抠图帧（点击跳转  |  黄框=当前，蓝框=待处理，绿框✓=已抠图）  "
            f"已完成 {done} / {n_sel}")

    def _update_sel_label(self):
        self.sel_label.setText(
            f"已选 {len(self.selected)} / {len(self.frames)} 帧（来自帧数选择）")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frames:
            self._resize_timer.start()
