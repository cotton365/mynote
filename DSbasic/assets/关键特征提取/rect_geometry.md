"""
矩形几何约束校验模块
=================

提供矩形几何验证工具：对边平行、邻边垂直、透视中心、环形区域校验。

函数:
    is_valid_rect(pts, min_side, angle_tol, opp_side_ratio) -> tuple[bool, str]
        校验四边形是否满足矩形几何约束
    perspective_center(pts) -> tuple[float, float]
        透视矫正后的四边形中心
    check_frame_ring(outer_pts, inner_pts, binary, center_dist_thresh, ring_ratio_thresh) -> tuple[bool, str]
        校验一对矩形是否构成黑框（环状区域白色占比 + 中心接近）

参数:
    各函数参数说明见具体 docstring。

来源:
    aim.py — 矩形检测与黑框配对逻辑中的几何校验函数
"""

import cv2
import math
import numpy as np


def _are_parallel(theta1: float, theta2: float, tolerance: float = 30) -> bool:
    """两个方向角是否平行（容差 ±tolerance 度）"""
    diff = abs(theta1 - theta2) % 180
    if diff > 90:
        diff = 180 - diff
    return diff <= tolerance


def _are_perpendicular(theta1: float, theta2: float, tolerance: float = 30) -> bool:
    """两个方向角是否垂直（容差 ±tolerance 度）"""
    diff = abs(theta1 - theta2) % 180
    if diff > 90:
        diff = 180 - diff
    return abs(diff - 90) <= tolerance


def perspective_center(pts: np.ndarray) -> tuple:
    """
    透视矫正后的四边形中心。

    将 4 个角点视作矩形的透视投影，通过单应变换矫正为正规矩形，
    取矫正后矩形的中心，再映射回原始图像坐标。

    参数:
        pts: (4, 2) array，四边形四个角点（按轮廓顺序）

    返回:
        (cx, cy): 透视矫正后的中心坐标 (float, float)
        退化时回退到质心
    """
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"perspective_center expected (4,2), got {pts.shape}")

    w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[3] - pts[2])) / 2.0
    h = (np.linalg.norm(pts[2] - pts[1]) + np.linalg.norm(pts[0] - pts[3])) / 2.0

    if w < 1e-6 or h < 1e-6:
        return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H_inv = cv2.getPerspectiveTransform(dst_pts, pts)
    center_dst = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
    center_src = cv2.perspectiveTransform(center_dst, H_inv)
    cx, cy = center_src[0, 0]
    return (float(cx), float(cy))


def is_valid_rect(pts: np.ndarray,
                  min_side: float = 20,
                  angle_tol: float = 30,
                  opp_side_ratio: float = 0.6) -> tuple:
    """
    校验四边形是否满足几何矩形约束。

    约束:
      1. 恰好 4 个顶点
      2. 每条边 > min_side px
      3. 对边平行（容差 ±angle_tol°）
      4. 邻边垂直（容差 ±angle_tol°）
      5. 对边长度比 > opp_side_ratio

    参数:
        pts: (N, 2) array，多边形顶点
        min_side: 最小边长 (px)
        angle_tol: 平行/垂直容差 (度)
        opp_side_ratio: 对边最小长度比

    返回:
        (is_valid: bool, detail: str)
    """
    if len(pts) != 4:
        return False, f"顶点数={len(pts)} ≠ 4"

    edges = []
    for i in range(4):
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[(i + 1) % 4], dtype=float)
        vec = p2 - p1
        length = np.linalg.norm(vec)
        edges.append({
            "vec": vec,
            "len": length,
            "theta": math.degrees(math.atan2(vec[1], vec[0])),
        })

    for i, e in enumerate(edges):
        if e["len"] <= min_side:
            return False, f"边{i}长度={e['len']:.1f} ≤ {min_side}"

    if not _are_parallel(edges[0]["theta"], edges[2]["theta"], angle_tol):
        return False, "对边 e0∦e2"
    if not _are_parallel(edges[1]["theta"], edges[3]["theta"], angle_tol):
        return False, "对边 e1∦e3"

    if not _are_perpendicular(edges[0]["theta"], edges[1]["theta"], angle_tol):
        return False, "邻边 e0⊥e1 不成立"

    ratio1 = min(edges[0]["len"], edges[2]["len"]) / max(edges[0]["len"], edges[2]["len"])
    ratio2 = min(edges[1]["len"], edges[3]["len"]) / max(edges[1]["len"], edges[3]["len"])
    if ratio1 < opp_side_ratio:
        return False, f"对边比1={ratio1:.2f} < {opp_side_ratio}"
    if ratio2 < opp_side_ratio:
        return False, f"对边比2={ratio2:.2f} < {opp_side_ratio}"

    return True, "OK"


def check_frame_ring(outer_pts: np.ndarray,
                     inner_pts: np.ndarray,
                     binary: np.ndarray,
                     center_dist_thresh: float = 15,
                     ring_ratio_thresh: float = 0.8) -> tuple:
    """
    校验一对矩形是否构成黑框：中心接近 + 环形区域白色占比 > ring_ratio_thresh。

    参数:
        outer_pts: 外矩形 (4, 2) float
        inner_pts: 内矩形 (4, 2) float
        binary: 二值图（白色=目标形状，黑色=背景）
        center_dist_thresh: 曼哈顿距离阈值（px）
        ring_ratio_thresh: 环形区域白色占比阈值

    返回:
        (is_valid: bool, detail: str)
    """
    # 1. 透视矫正中心
    ocx, ocy = perspective_center(outer_pts)
    icx, icy = perspective_center(inner_pts)

    # 2. 曼哈顿距离
    dist = abs(ocx - icx) + abs(ocy - icy)
    if dist >= center_dist_thresh:
        return False, f"中心距离={dist:.1f} >= {center_dist_thresh}"

    # 3. 确认外矩形面积 >= 内矩形面积（必要时交换）
    outer_area = cv2.contourArea(outer_pts.astype(np.float32))
    inner_area = cv2.contourArea(inner_pts.astype(np.float32))
    if outer_area < inner_area:
        outer_pts, inner_pts = inner_pts.copy(), outer_pts.copy()
        outer_area, inner_area = inner_area, outer_area

    # 4. 验证内矩形四个角点都在外矩形内部
    outer_i32 = outer_pts.astype(np.int32)
    inner_i32 = inner_pts.astype(np.int32)
    for corner in inner_i32:
        if cv2.pointPolygonTest(outer_i32, (float(corner[0]), float(corner[1])), False) < -1:
            return False, "内矩形角点在外矩形外部"

    # 5. 环形区域（外矩形内、内矩形外）白色占比
    mask_outer = np.zeros(binary.shape, dtype=np.uint8)
    mask_inner = np.zeros(binary.shape, dtype=np.uint8)
    cv2.fillPoly(mask_outer, [outer_i32], 255)
    cv2.fillPoly(mask_inner, [inner_i32], 255)

    ring_mask = cv2.subtract(mask_outer, mask_inner)
    ring_total = cv2.countNonZero(ring_mask)
    if ring_total == 0:
        return False, "环形区域面积为0"

    ring_white = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=ring_mask))
    ratio = ring_white / ring_total

    if ratio > ring_ratio_thresh:
        return True, f"OK ring_ratio={ratio:.3f}"
    else:
        return False, f"环形白色占比={ratio:.3f} <= {ring_ratio_thresh}"
