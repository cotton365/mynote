"""
现实尺寸换算模块
============

将轮廓像素面积通过面积比系数换算为实际物理尺寸 (mm)。

函数:
    compute_real_dimension(contour, shape_type, area_ratio) -> tuple[float, float]
        像素面积 → 现实尺寸

参数:
    contour : np.ndarray
        OpenCV 轮廓 (N, 1, 2)
    shape_type : str
        形状类型: "circle" | "square" | "triangle" | "hexagon"
    area_ratio : float
        现实面积(mm²) / 像素面积(px²)，由 A4CalibEstimator.compute_area_ratio() 得到

返回:
    tuple[float, float]:
        (real_dimension_mm, real_area_mm2)
        circle → 直径, 正方形 → 边长, 三角形/六边形 → 边长
"""

import cv2
import math


def compute_real_dimension(contour: np.ndarray,
                           shape_type: str,
                           area_ratio: float) -> tuple:
    """
    把轮廓像素面积换算为实际物理尺寸。

    参数:
        contour: OpenCV 轮廓 (N, 1, 2)
        shape_type: "circle" | "square" | "triangle" | "hexagon" | "unknown"
        area_ratio: 面积比 (mm²/px²)

    返回:
        (real_dimension_mm: float, real_area_mm2: float)
        circle → 返回直径, 其余正多边形 → 返回边长
        未知类型返回 (0.0, real_area)
    """
    pixel_area = cv2.contourArea(contour)
    real_area = pixel_area * area_ratio

    if shape_type == "circle":
        diameter = 2 * math.sqrt(real_area / math.pi)
        return diameter, real_area
    elif shape_type == "square":
        side = math.sqrt(real_area)
        return side, real_area
    elif shape_type == "triangle":
        side = math.sqrt(4 * real_area / math.sqrt(3))
        return side, real_area
    elif shape_type == "hexagon":
        side = math.sqrt(2 * real_area / (3 * math.sqrt(3)))
        return side, real_area
    else:
        return 0.0, real_area
