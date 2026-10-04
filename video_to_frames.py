# -*- coding: utf-8 -*-
"""
视频导出为序列帧模块

功能：
1. 从视频中按指定策略抽取帧（逐帧 / 每 N 帧 / 每 N 秒）
2. 相同/相似帧去重（基于感知哈希 pHash，阈值可调）
3. 保存帧图片，并生成 frames_info.json 供后续步骤使用

用法：
    python video_to_frames.py input.mp4 -o frames --mode interval --interval 5 --dedup
    python video_to_frames.py input.mp4 -o frames --mode time --interval 0.5 --dedup --threshold 95
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ===================== 数据结构 =====================

@dataclass
class FrameInfo:
    """单帧信息记录"""
    index: int           # 导出后的序号（从0开始）
    original_index: int  # 原始视频中的帧号（从0开始）
    timestamp_ms: float  # 时间戳（毫秒）
    filename: str        # 保存的文件名
    hash: str            # 该帧的感知哈希（64位二进制字符串）
    is_duplicate: bool   # 是否因重复而被跳过


# ===================== 感知哈希 =====================

def perceptual_hash(frame_bgr: np.ndarray, hash_size: int = 16) -> str:
    """
    计算图像的感知哈希 (pHash)。
    返回 hash_size*hash_size 位的二进制字符串。
    算法：灰度化 -> resize -> DCT -> 取左上低频 -> 比较中位数
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # 放大一点再做 DCT，效果更好
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    resized = np.float32(resized)

    dct = cv2.dct(resized)
    # 取左上角低频区域（去掉直流分量）
    dct_low = dct[1:hash_size, 1:hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    return ''.join('1' if b else '0' for b in bits)


def hamming_distance(hash1: str, hash2: str) -> int:
    """两个哈希串的汉明距离（不同的位数）"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def similarity_percent(hash1: str, hash2: str) -> float:
    """
    相似度百分比 (0~100)。
    100 = 完全相同，0 = 完全不同。
    """
    total = len(hash1)
    if total == 0:
        return 0.0
    dist = hamming_distance(hash1, hash2)
    return (1.0 - dist / total) * 100.0


# ===================== 抽帧核心 =====================

def extract_frames(
    video_path: str,
    output_dir: str,
    mode: str = "all",            # all | interval | time
    interval: float = 1,          # 帧间隔(整数) 或 秒间隔(浮点)
    dedup: bool = False,
    similarity_threshold: float = 95.0,  # 相似度>=此值判定为重复
    img_format: str = "png",      # png | jpg
    prefix: str = "frame",
    start_time: float = 0.0,      # 起始时间(秒)
    end_time: Optional[float] = None,  # 结束时间(秒)
    jpg_quality: int = 95,
) -> List[FrameInfo]:
    """
    从视频抽取帧并保存。

    返回导出的帧信息列表（仅包含实际保存的帧，不含被去重跳过的）。
    同时在 output_dir 下生成 frames_info.json，记录所有处理过的帧（含被跳过的）。
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 清理同前缀的旧帧图片和旧数据文件，避免残留导致后续模块帧数对不上
    old_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for fn in os.listdir(output_dir):
        fn_lower = fn.lower()
        is_old_frame = (
            fn_lower.startswith(prefix.lower() + "_")
            and fn_lower.endswith(old_exts))
        is_old_json = fn in ("frames_info.json", "selected_frames.json")
        if is_old_frame or is_old_json:
            try:
                os.remove(os.path.join(output_dir, fn))
            except OSError:
                pass

    # 清理下游模块产物（批量抠图 matted/、图片精抠 refined/、色调统一 colored/），
    # 否则换新视频重新抽帧后，旧视频的产物会残留并被下游页面读到
    for sub in ("matted", "refined", "colored"):
        sub_dir = os.path.join(output_dir, sub)
        if os.path.isdir(sub_dir):
            shutil.rmtree(sub_dir, ignore_errors=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    # 时间范围换算为帧号
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else total_frames

    # 计算目标帧集合
    target_indices: List[int] = []
    if mode == "all":
        target_indices = list(range(start_frame, end_frame))
    elif mode == "interval":
        step = max(1, int(interval))
        target_indices = list(range(start_frame, end_frame, step))
    elif mode == "time":
        step_frames = max(1, int(round(interval * fps)))
        target_indices = list(range(start_frame, end_frame, step_frames))
    else:
        cap.release()
        raise ValueError(f"未知的抽帧模式: {mode}（支持 all / interval / time）")

    print(f"[信息] 视频 FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}s")
    print(f"[信息] 抽帧模式={mode}, 待处理目标帧数={len(target_indices)}")
    if dedup:
        print(f"[信息] 已启用去重，相似度阈值>={similarity_threshold}% 视为重复")

    saved_frames: List[FrameInfo] = []
    all_records: List[FrameInfo] = []

    export_index = 1
    last_hash: Optional[str] = None

    for idx, frame_no in enumerate(target_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[警告] 第 {frame_no} 帧读取失败，跳过")
            continue

        timestamp_ms = frame_no / fps * 1000.0
        cur_hash = perceptual_hash(frame)

        is_dup = False
        if dedup and last_hash is not None:
            sim = similarity_percent(cur_hash, last_hash)
            if sim >= similarity_threshold:
                is_dup = True

        filename = f"{prefix}_{export_index:05d}.{img_format}"
        filepath = os.path.join(output_dir, filename)

        if is_dup:
            # 不保存图片，但记录信息
            record = FrameInfo(
                index=-1,
                original_index=frame_no,
                timestamp_ms=round(timestamp_ms, 2),
                filename="",
                hash=cur_hash,
                is_duplicate=True,
            )
            all_records.append(record)
        else:
            # 保存图片
            if img_format == "jpg":
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            else:
                cv2.imwrite(filepath, frame)  # PNG 默认无损

            record = FrameInfo(
                index=export_index,
                original_index=frame_no,
                timestamp_ms=round(timestamp_ms, 2),
                filename=filename,
                hash=cur_hash,
                is_duplicate=False,
            )
            saved_frames.append(record)
            all_records.append(record)
            last_hash = cur_hash
            export_index += 1

        # 进度提示（每帧更新，便于 GUI 进度条实时显示）
        # export_index 是下一个待保存的序号（从1开始），已保存数 = export_index - 1
        print(f"[进度] {idx + 1}/{len(target_indices)} "
              f"已保存 {export_index - 1} 帧", flush=True)

    cap.release()

    # 写入 frames_info.json
    info_path = os.path.join(output_dir, "frames_info.json")
    info_data = {
        "video_path": os.path.abspath(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration,
        "mode": mode,
        "interval": interval,
        "dedup_enabled": dedup,
        "similarity_threshold": similarity_threshold,
        "exported_count": export_index - 1,
        "skipped_duplicates": len(target_indices) - (export_index - 1),
        "frames": [asdict(r) for r in all_records],
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_data, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 共处理 {len(target_indices)} 帧")
    print(f"[完成] 实际保存 {export_index - 1} 帧，跳过重复 {len(target_indices) - (export_index - 1)} 帧")
    print(f"[完成] 输出目录: {os.path.abspath(output_dir)}")
    print(f"[完成] 帧信息文件: {os.path.abspath(info_path)}")

    return saved_frames


# ===================== 命令行入口 =====================

def main():
    # 不带任何参数运行时，直接打开主应用（集成所有功能模块）
    if len(sys.argv) == 1:
        try:
            from main_app import main as app_main
            app_main()
            return
        except ImportError:
            print("[提示] 未找到主应用模块，请传入命令行参数或运行 main_app.py",
                  file=sys.stderr)

    parser = argparse.ArgumentParser(
        description="视频导出为序列帧（支持去重）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("-o", "--output", default="frames", help="输出目录（默认: frames）")
    parser.add_argument(
        "--mode",
        choices=["all", "interval", "time"],
        default="all",
        help="抽帧模式: all=逐帧, interval=每N帧, time=每N秒（默认: all）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1,
        help="间隔帧数(mode=interval)或间隔秒数(mode=time)（默认: 1）",
    )
    parser.add_argument("--dedup", action="store_true", help="启用相同帧去重")
    parser.add_argument(
        "--threshold",
        type=float,
        default=95.0,
        help="去重相似度阈值 0~100，>=此值视为重复（默认: 95）",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="输出图片格式（默认: png 无损）",
    )
    parser.add_argument("--prefix", default="frame", help="文件名前缀（默认: frame）")
    parser.add_argument("--start", type=float, default=0.0, help="起始时间（秒）")
    parser.add_argument("--end", type=float, default=None, help="结束时间（秒）")

    args = parser.parse_args()

    try:
        extract_frames(
            video_path=args.video,
            output_dir=args.output,
            mode=args.mode,
            interval=args.interval,
            dedup=args.dedup,
            similarity_threshold=args.threshold,
            img_format=args.format,
            prefix=args.prefix,
            start_time=args.start,
            end_time=args.end,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
