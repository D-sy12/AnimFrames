# -*- coding: utf-8 -*-
"""
模块六：导出图片 - 核心算法（无 GUI 依赖，可独立测试）

功能一：动画精灵图（sprite sheet）
  build_sprite_sheet() 把一批带透明通道的帧按行列网格拼在一张透明底
  PNG 上，支持列数（0=自动近方形）、每帧缩放（调分辨率）、单元格外边距。

功能二：PSD 分层图（类似 Photoshop 的图层堆叠）
  write_layered_psd() 把每帧写成 PSD 的一个图层，各图层可独立缩放、
  独立定位，底色透明。写入器自包含（仅 numpy + 标准库），8 位 RGB+
  透明 alpha，通道数据采用 PackBits（RLE）压缩，Photoshop / Krita /
  Clip Studio / Photopea 均可打开；为兼容旧版 PS，图层名同时写入
  Pascal 名与 luni(UTF-16BE) 标记块，中文名正常显示。
"""

import os
import struct
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2

from gui_frame_matting import imread_unicode, imwrite_unicode


# ===================== 通用工具 =====================

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
    """只读图片宽高，不解码整幅（用 PIL 解析头部）；失败则回退 cv2"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        pass
    img = imread_unicode(path)
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def scale_bgra(img: np.ndarray, scale: float) -> np.ndarray:
    """等比缩放 BGRA（含 alpha 双线性/面积插值，最小 1px）"""
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if nw == w and nh == h:
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def flatten_over(canvas: np.ndarray, layer: np.ndarray,
                 x: int, y: int) -> np.ndarray:
    """把 BGRA 图层以 over 合成方式贴到画布的 (x,y)，支持越界裁剪"""
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
    # 标准 straight-alpha over：先做预乘累积，再除回总覆盖率
    ao = sa + da * (1.0 - sa)
    prem = sub[:, :, :3] * sa + dst[:, :, :3] * da * (1.0 - sa)
    safe = np.clip(ao, 1e-6, None)
    out = np.empty_like(dst)
    out[:, :, :3] = np.where(ao > 1e-6, prem / safe, 0.0)
    out[:, :, 3:4] = ao * 255.0
    canvas[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return canvas


# ===================== 精灵图拼接 =====================

def build_sprite_sheet(images: List[np.ndarray],
                       columns: int = 0,
                       scale: float = 1.0,
                       padding: int = 0
                       ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    把多张 BGRA 帧拼成一张透明底精灵图。
    images    : 各帧（允许尺寸不一致，按各自尺寸*scale 缩放后居中于单元格）
    columns   : 每行帧数；0 = 自动近方形（列数=ceil(sqrt(n))）
    scale     : 每帧缩放比例（调导出分辨率）
    padding   : 单元格外边距/间距像素
    返回 (画布 BGRA, 每帧在画布上的矩形 [(x,y,w,h), ...])
    """
    n = len(images)
    if n == 0:
        raise ValueError("没有可拼接的图片")
    cols = int(columns) if columns and columns > 0 else \
        int(np.ceil(np.sqrt(n)))
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))

    scaled = [scale_bgra(im, scale) for im in images]
    cell_w = max(im.shape[1] for im in scaled)
    cell_h = max(im.shape[0] for im in scaled)

    W = cols * cell_w + (cols + 1) * padding
    H = rows * cell_h + (rows + 1) * padding
    canvas = np.zeros((H, W, 4), np.uint8)

    rects: List[Tuple[int, int, int, int]] = []
    for i, im in enumerate(scaled):
        r, c = divmod(i, cols)
        x = padding + c * (cell_w + padding)
        y = padding + r * (cell_h + padding)
        ox = x + (cell_w - im.shape[1]) // 2
        oy = y + (cell_h - im.shape[0]) // 2
        flatten_over(canvas, im, ox, oy)
        rects.append((ox, oy, im.shape[1], im.shape[0]))
    return canvas, rects


def sprite_sheet_size(sizes: List[Tuple[int, int]],
                      columns: int = 0,
                      scale: float = 1.0,
                      padding: int = 0) -> Tuple[int, int, int, int]:
    """不实际拼接，快速推算输出尺寸与网格 (W,H,cols,rows)"""
    n = len(sizes)
    if n == 0:
        return 0, 0, 0, 0
    cols = int(columns) if columns and columns > 0 else \
        int(np.ceil(np.sqrt(n)))
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    cw = max(int(round(w * scale)) for w, h in sizes)
    ch = max(int(round(h * scale)) for w, h in sizes)
    W = cols * cw + (cols + 1) * padding
    H = rows * ch + (rows + 1) * padding
    return W, H, cols, rows


# ===================== PSD（分层图）写入 =====================

def _packbits_encode(data: bytes) -> bytes:
    """PackBits 压缩（PSD 单通道 8bit 用），大量透明零压缩率很高"""
    out = bytearray()
    n = len(data)
    i = 0
    while i < n:
        # 向后探测连续相同字节
        j = i + 1
        while j < n and j - i < 128 and data[j] == data[i]:
            j += 1
        run = j - i
        if run >= 3:
            out.append(257 - run)          # -(run-1) 的无符号表示
            out.append(data[i])
            i = j
            continue
        # 字面值序列：遇到下一段 >=3 重复或满 128 字节为止
        lit = bytearray()
        while i < n and len(lit) < 128:
            if i + 2 < n and data[i] == data[i + 1] == data[i + 2]:
                break
            lit.append(data[i])
            i += 1
        out.append(len(lit) - 1)
        out.extend(lit)
    return bytes(out)


def _packbits_rows(plane: np.ndarray) -> List[bytes]:
    """逐扫描行做 PackBits，返回每行压缩结果（PSD RLE 以行为单位）"""
    return [_packbits_encode(np.ascontiguousarray(plane[y]).tobytes())
            for y in range(plane.shape[0])]


def _rle_channel_blob(plane: np.ndarray) -> bytes:
    """单个通道的完整 RLE 数据：h 个 uint16 行长度 + 各行压缩数据"""
    rows = _packbits_rows(plane)
    counts = b"".join(struct.pack(">H", len(r)) for r in rows)
    return counts + b"".join(rows)


def _psd_pascal_name(name: str) -> bytes:
    """Pascal 风格图层名（1 字节长度+Latin1），补齐到 4 字节倍数"""
    raw = name.encode("latin-1", "replace")[:255]
    p = bytes([len(raw)]) + raw
    while len(p) % 4:
        p += b"\x00"
    return p


def _psd_layer_record(top: int, left: int, bottom: int, right: int,
                      plane_blobs: List[bytes], name: str,
                      visible: bool = True) -> bytes:
    ch_info = b""
    for cid, blob in zip((0, 1, 2, -1), plane_blobs):   # R,G,B,透明
        ch_info += struct.pack(">hi", cid, 2 + len(blob))

    # luni 标记块：UTF-16BE 图层名，保证中文正常
    u16 = name.encode("utf-16-be")
    luni = struct.pack(">I", len(name)) + u16
    if len(luni) % 2:
        luni += b"\x00"
    tagged = b"8BIMluni" + struct.pack(">I", len(luni)) + luni

    # 附加数据：蒙版长度0 + 混合范围长度0 + 名字 + 标记块
    extra_body = (struct.pack(">I", 0) + struct.pack(">I", 0)
                  + _psd_pascal_name(name) + tagged)
    extra = struct.pack(">I", len(extra_body)) + extra_body

    rec = struct.pack(">iiii", top, left, bottom, right)
    rec += struct.pack(">H", 4) + ch_info
    # flags 第2位(bit1)=隐藏：不透明255, 锁定0, flags, 填充0
    flags = 0 if visible else 0x02
    rec += b"8BIMnorm" + bytes((255, 0, flags, 0)) + extra
    return rec


def write_layered_psd(path: str,
                      canvas_w: int,
                      canvas_h: int,
                      layers: List[Tuple],
                      composite: Optional[np.ndarray] = None) -> str:
    """
    写 8 位 RGB+Alpha 的分层 PSD。
    layers : [(图层名, BGRA 图, x, y[, visible]), ...]，
             x/y 为左上角，可越界/负坐标；visible=False 的图层在 PSD 里
             为隐藏状态，且不参与自动合并预览。
    composite: 可选合并预览图（BGRA，画布尺寸）；不给定则自动 over 合成
    """
    if not layers:
        raise ValueError("PSD 至少需要一个图层")

    records = bytearray()
    all_channel_data = bytearray()
    for layer in layers:
        name, img, x, y = layer[0], layer[1], layer[2], layer[3]
        visible = bool(layer[4]) if len(layer) > 4 else True
        if img.ndim != 3 or img.shape[2] != 4:
            raise ValueError(f"图层 {name} 必须是 BGRA 四通道")
        h, w = img.shape[:2]
        planes = (np.ascontiguousarray(img[:, :, 2]),   # R
                  np.ascontiguousarray(img[:, :, 1]),   # G
                  np.ascontiguousarray(img[:, :, 0]),   # B
                  np.ascontiguousarray(img[:, :, 3]))   # A
        blobs = [_rle_channel_blob(p) for p in planes]
        records += _psd_layer_record(y, x, y + h, x + w, blobs, name,
                                     visible)
        for blob in blobs:
            all_channel_data += struct.pack(">H", 1) + blob   # 1=PackBits

    # 图层信息：层数(可负) + 记录 + 通道数据，整体偶对齐
    inner = struct.pack(">h", len(layers)) + bytes(records) + \
        bytes(all_channel_data)
    if len(inner) % 2:
        inner += b"\x00"
    layer_info = struct.pack(">I", len(inner)) + inner
    global_mask = struct.pack(">I", 0)
    section = layer_info + global_mask

    # 合并预览（composite image data）
    if composite is None:
        composite = np.zeros((canvas_h, canvas_w, 4), np.uint8)
        for layer in layers:
            if len(layer) > 4 and not layer[4]:
                continue                       # 隐藏图层不进合并预览
            flatten_over(composite, layer[1], layer[2], layer[3])
    comp_planes = (np.ascontiguousarray(composite[:, :, 2]),  # R
                   np.ascontiguousarray(composite[:, :, 1]),  # G
                   np.ascontiguousarray(composite[:, :, 0]),  # B
                   np.ascontiguousarray(composite[:, :, 3]))  # A
    # PackBits：先给全部通道的逐行计数（4*h 个 uint16），再放压缩数据
    comp_rows = [_packbits_rows(p) for p in comp_planes]
    counts = b"".join(
        struct.pack(">H", len(r)) for rows in comp_rows for r in rows)
    packets = b"".join(r for rows in comp_rows for r in rows)
    image_data = struct.pack(">H", 1) + counts + packets

    header = b"8BPS" + struct.pack(">H", 1) + b"\x00" * 6
    header += struct.pack(">HIIHH", 4, canvas_h, canvas_w, 8, 3)
    color_mode = struct.pack(">I", 0)
    resources = struct.pack(">I", 0)
    mask_section = struct.pack(">I", len(section)) + section

    data = header + color_mode + resources + mask_section + image_data
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def centered_xy(canvas_w: int, canvas_h: int,
                w: int, h: int) -> Tuple[int, int]:
    return (canvas_w - w) // 2, (canvas_h - h) // 2
