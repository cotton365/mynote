"""
图像预处理模块
===========

对 BGR 图像执行预处理管线: 可选白平衡校正 → CLAHE 对比度增强 → 高斯模糊。

函数:
    preprocess(frame: np.ndarray) -> np.ndarray
        对输入帧执行预处理，返回处理后的 BGR 图像。

参数:
    frame : np.ndarray
        输入 BGR 图像 (H, W, 3), dtype=uint8

返回:
    np.ndarray: 预处理后的 BGR 图像，尺寸与输入相同

依赖:
    camera_config.py — 读取 preprocessing 配置段 (clahe_clip, clahe_tile, blur_ksize, wb_correction, wb_bgr_gain)
"""

import cv2
import numpy as np
from camera_config import load_config


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    预处理管线: LAB→CLAHE(L通道)→BGR→GaussianBlur。

    流程:
        1. 可选白平衡: frame = clip(frame * wb_bgr_gain)
        2. BGR→LAB, CLAHE 仅对 L 通道
        3. LAB→BGR
        4. GaussianBlur (ksize > 1 时)

    参数:
        frame: BGR 图像 (H, W, 3), uint8

    返回:
        预处理后的 BGR 图像 (H, W, 3), uint8
    """
    cfg = load_config()
    pp = cfg["preprocessing"]

    if pp.get("wb_correction", False):
        gain = np.array(pp["wb_bgr_gain"], dtype=np.float32)
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=pp["clahe_clip"],
        tileGridSize=(pp["clahe_tile"], pp["clahe_tile"])
    )
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    frame = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    ks = pp["blur_ksize"]
    if ks > 1:
        frame = cv2.GaussianBlur(frame, (ks, ks), 0)

    return frame
