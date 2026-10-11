"""
物体可视化模块
===========

在图像上叠加物体标签（形状类型 + 像素/现实尺寸）。

函数:
    ObjectShow(objects, frame, area_ratio=None) -> np.ndarray
        在图像上绘制每个物体的标签，返回标注副本。

参数:
    objects : list[ShapeObject]
        物体列表 (来自 object_extractor.get_objects)
    frame : np.ndarray
        待标注的 BGR 图像 (H, W, 3) 或灰度图
    area_ratio : float | None
        现实面积比 (mm²/px²)。若为 None，显示像素尺寸；否则显示 mm 尺寸

返回:
    np.ndarray: 标注后的 BGR 图像 (H, W, 3)

依赖:
    dimension_calculator.py — compute_real_dimension()
"""

import cv2
from dimension_calculator import compute_real_dimension


def ObjectShow(objects: list, frame, area_ratio: float = None):
    """
    在图像上标注每个物体的形状类型和尺寸。

    显示规则:
        - 有 area_ratio: circle→"d=XX.Xmm", 其他→"s=XX.Xmm"
        - 无 area_ratio: 显示 "shape X.Xpx"

    参数:
        objects: ShapeObject 列表
        frame: BGR或灰度图像 (H, W, 3) 或 (H, W)
        area_ratio: 面积比 (mm²/px²)，None 则显示像素尺寸

    返回:
        np.ndarray: BGR 标注图像（不修改原图）
    """
    processed_frame = frame.copy()

    if len(processed_frame.shape) == 2:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    for obj in objects:
        if obj.center is None:
            continue
        x, y = obj.center

        if area_ratio is not None and area_ratio > 0:
            real_dim, real_area = compute_real_dimension(obj.contour, obj.shape_type, area_ratio)
            if obj.shape_type == "circle":
                text = f"{obj.shape_type} d={real_dim:.1f}mm"
            else:
                text = f"{obj.shape_type} s={real_dim:.1f}mm"
        else:
            text = f"{obj.shape_type} {obj.size_px:.1f}px"

        cv2.putText(processed_frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    return processed_frame
