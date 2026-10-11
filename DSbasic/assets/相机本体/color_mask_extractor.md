"""
颜色遮罩提取模块
============

从 BGR 图像提取红/绿/蓝三色 HSV 遮罩，经过形态学开闭运算清理噪点。

函数:
    extract_masks(frame_bgr: np.ndarray) -> dict
        提取三色遮罩，返回字典 {"red": mask, "green": mask, "blue": mask}

参数:
    frame_bgr : np.ndarray
        BGR 图像 (H, W, 3), uint8

返回:
    dict: {"red": np.ndarray, "green": np.ndarray, "blue": np.ndarray}
        每个遮罩为二值图 (H, W), uint8, 0=背景 255=目标

依赖:
    camera_config.py — 读取 hsv 和 morphology 配置段
"""

import cv2
import numpy as np
from camera_config import load_config


def extract_masks(frame_bgr: np.ndarray) -> dict:
    """
    从 BGR 帧提取红/绿/蓝三色 HSV 遮罩。

    流程:
        1. BGR→HSV
        2. 红(两段 wraparound 取 or)/绿/蓝 分别 cv2.inRange
        3. 形态学 OPEN + CLOSE (椭圆形核)

    参数:
        frame_bgr: BGR 图像 (H, W, 3), uint8

    返回:
        {"red": mask, "green": mask, "blue": mask}
        每个 mask 为二值 uint8, 255=目标区域
    """
    cfg = load_config()
    h = cfg["hsv"]
    mo = cfg["morphology"]

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    def make_range(key):
        d = h[key]
        lo = np.array([d["h_lo"], d["s_lo"], d["v_lo"]], dtype=np.uint8)
        hi = np.array([d["h_hi"], d["s_hi"], d["v_hi"]], dtype=np.uint8)
        return lo, hi

    lo1, hi1 = make_range("red1")
    lo2, hi2 = make_range("red2")
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lo1, hi1),
        cv2.inRange(hsv, lo2, hi2)
    )

    lo_g, hi_g = make_range("green")
    mask_green = cv2.inRange(hsv, lo_g, hi_g)

    lo_b, hi_b = make_range("blue")
    mask_blue = cv2.inRange(hsv, lo_b, hi_b)

    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mo["open_ksize"],  mo["open_ksize"]))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mo["close_ksize"], mo["close_ksize"]))

    def clean(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        return mask

    return {
        "red":   clean(mask_red),
        "green": clean(mask_green),
        "blue":  clean(mask_blue),
    }
