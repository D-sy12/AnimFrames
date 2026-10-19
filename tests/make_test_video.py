# -*- coding: utf-8 -*-
"""生成测试视频：包含静止帧和变化帧，用于验证抽帧与去重功能"""
import os

import cv2
import numpy as np

W, H = 320, 240
FPS = 10
TOTAL = 60  # 6秒

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

for i in range(TOTAL):
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    if i < 20:
        # 前2秒：红色方块静止（测试去重）
        color = (0, 0, 255)
    elif i < 40:
        # 中间2秒：绿色方块，每秒移动一点
        color = (0, 255, 0)
    else:
        # 最后2秒：蓝色方块，每帧颜色渐变
        color = (255, int(255 * i / TOTAL), 0)

    x = 50 + (i % 10) * 10 if 20 <= i < 40 else 100
    cv2.rectangle(frame, (x, 80), (x + 80, 160), color, -1)
    cv2.putText(frame, f"frame {i}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out.write(frame)

out.release()
print(f"测试视频已生成: {OUT} (6秒, 10fps, 60帧)")
