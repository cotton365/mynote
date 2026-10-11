"""
亮蓝目标检测模块
=============

HSV 遮罩 + 形态学处理，从 BGR 图像中隔离"非常亮蓝色"目标区域。

函数:
    extract_bright_blue_mask(frame_bgr: np.ndarray, config: dict) -> np.ndarray
        HSV 阈值 → dilate → erode → 输出二值遮罩

参数:
    frame_bgr : np.ndarray
        预处理后的 BGR 图像 (H, W, 3), uint8
    config : dict
        包含 "hsv" 和 "morphology" 段的配置字典

返回:
    np.ndarray: 二值遮罩 (H, W), uint8, 255=亮蓝目标区域

来源:
    blue_white.py — extract_target_mask() 函数
"""

import cv2
import numpy as np


def extract_bright_blue_mask(frame_bgr: np.ndarray, config: dict) -> np.ndarray:
    """
    从 BGR 图像提取亮蓝色目标二值遮罩。

    流程:
        1. BGR → HSV
        2. cv2.inRange 根据 hsv.target 阈值过滤
        3. dilate (填充孔洞)
        4. erode (修剪边缘噪声)

    参数:
        frame_bgr: BGR 图像 (H, W, 3), uint8
        config: 配置字典，需要包含:
            config["hsv"]["target"]: {"h_lo", "h_hi", "s_lo", "s_hi", "v_lo", "v_hi"}
            config["morphology"]:    {"dilate_iter", "erode_iter", "kernel_size"}

    返回:
        np.ndarray: 二值遮罩 (H, W), uint8
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = config["hsv"]["target"]
    mo = config["morphology"]

    lo = np.array([h["h_lo"], h["s_lo"], h["v_lo"]], dtype=np.uint8)
    hi = np.array([h["h_hi"], h["s_hi"], h["v_hi"]], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    ks = mo["kernel_size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

    if mo["dilate_iter"] > 0:
        mask = cv2.dilate(mask, kernel, iterations=mo["dilate_iter"])
    if mo["erode_iter"] > 0:
        mask = cv2.erode(mask, kernel, iterations=mo["erode_iter"])

    return mask
