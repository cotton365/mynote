"""
白平衡标定模块
==========

从含白色区域的图像自动计算 RGB 白平衡增益系数。

函数:
    calibrate_wb_from_white(frame: np.ndarray, config: dict) -> dict
        检测白色区域，计算 BGR 增益，写回 config 字典。

参数:
    frame : np.ndarray
        待标定的 BGR 图像 (H, W, 3), uint8
    config : dict
        包含 preprocessing.wb_bgr_gain 段的配置字典（原地修改）

返回:
    dict: 修改后的 config 字典（同一对象）
"""

import cv2
import numpy as np


def calibrate_wb_from_white(frame: np.ndarray, config: dict) -> dict:
    """
    从白色区域自动标定白平衡增益。

    算法:
        1. 灰度图取 80 分位作为白色阈值
        2. 阈值以上区域视为白色，计算 B/G/R 均值
        3. 以 G 通道为基准，计算 B/R 增益

    参数:
        frame: BGR 图像 (H, W, 3), uint8
        config: 包含 "preprocessing" 键的配置字典
               修改 config["preprocessing"]["wb_bgr_gain"] 原地

    返回:
        修改后的 config 字典
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_val = np.percentile(gray, 80)
    _, mask = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)

    b_mean = np.mean(frame[:, :, 0][mask > 0])
    g_mean = np.mean(frame[:, :, 1][mask > 0])
    r_mean = np.mean(frame[:, :, 2][mask > 0])

    ref = g_mean
    gain_b = ref / b_mean if b_mean > 1 else 1.0
    gain_g = 1.0
    gain_r = ref / r_mean if r_mean > 1 else 1.0

    config["preprocessing"]["wb_bgr_gain"] = [
        round(float(gain_b), 4),
        round(float(gain_g), 4),
        round(float(gain_r), 4)
    ]
    print(f"[WB] White balance gain B={gain_b:.3f} G={gain_g:.3f} R={gain_r:.3f}")
    return config
