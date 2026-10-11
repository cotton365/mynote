"""
物体提取模块
=========

从二值遮罩中提取连通域，对每个轮廓进行形状分类，返回 ShapeObject 列表。

数据类:
    ShapeObject
        contour: np.ndarray — 轮廓 (N, 1, 2)
        shape_type: str — 形状类型
        size_px: float — 尺寸 (circle→半径, square→短边, 其他→包围圆半径)
        center: tuple[int, int] | None — 质心坐标 (x, y)

函数:
    get_objects(mask: np.ndarray) -> list[ShapeObject]
        找外轮廓 → 逐个分类 → 返回对象列表

参数:
    mask : np.ndarray
        二值遮罩 (H, W), uint8, 255=目标, 0=背景

返回:
    list[ShapeObject]: 检测到的物体列表

依赖:
    shape_classifier.py — classify_object()
"""

import cv2
import numpy as np
from dataclasses import dataclass
from shape_classifier import classify_object


@dataclass
class ShapeObject:
    """单个检测物体的属性。"""
    contour: np.ndarray
    shape_type: str
    size_px: float  # circle: 半径, polygon: 平均边长
    center: tuple = None  # (x, y)，可选


def get_objects(mask: np.ndarray) -> list:
    """
    从遮罩提取所有外轮廓，逐个分类。

    参数:
        mask: 二值遮罩 (H, W), uint8, 255=前景

    返回:
        list[ShapeObject]: 检测到的物体，每个包含轮廓、类型、尺寸、质心
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        shape_type, size_px = classify_object(cnt)

        M = cv2.moments(cnt)
        center = None
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

        obj = ShapeObject(contour=cnt, shape_type=shape_type, size_px=size_px, center=center)
        objects.append(obj)

    return objects
