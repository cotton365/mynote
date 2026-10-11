"""
正方形查找模块
============

从二值图中检测角点，筛选正方形候选顶点，构建并验证正方形，返回最小正方形。

函数:
    check_edge_type(binary, c1, c2, ...) -> tuple[str, np.ndarray | None]
        检查边的内外属性（EXPOSED/INTERNAL）
    filter_square_vertices(binary, all_corners, angle_thresh) -> list
        筛选可能为正方形顶点的角点
    build_square_from_edge(c1, c2, inward_normal) -> np.ndarray
        由暴露边和朝内法向量构建正方形四顶点
    validate_square_by_area_sampling(binary, square_pts, ...) -> bool
        正方形内部随机采样验证是否是实心目标
    are_squares_similar(sq1, sq2, ...) -> bool
        判断两个正方形是否为重复检测
    find_smallest_square(binary, valid_corners) -> np.ndarray | None
        主入口：寻找面积最小的真实正方形

来源:
    black.py — A4 纸内黑色正方形检测管线
"""

import cv2
import numpy as np
from itertools import combinations


def check_edge_type(binary: np.ndarray,
                    c1, c2,
                    sample_count: int = 20,
                    side_offset: int = 8,
                    white_ratio_thresh: float = 0.7) -> tuple:
    """
    检查边 c1->c2 是 EXPOSED（一侧前景/一侧背景）还是 INTERNAL。

    参数:
        binary: 二值图像 (白色=前景, 黑色=背景)
        c1, c2: 边的两个端点 (x, y)
        sample_count: 沿边采样点数
        side_offset: 边两侧偏移像素数
        white_ratio_thresh: 前景侧白色占比阈值

    返回:
        (edge_type: str, inward_normal: np.ndarray | None)
        edge_type: "EXPOSED" | "INTERNAL" | "UNKNOWN"
        inward_normal: 指向物体内部的单位向量（仅 EXPOSED 有效）
    """
    h, w = binary.shape[:2]
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    edge_vec = c2 - c1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-6:
        return "UNKNOWN", None

    edge_unit  = edge_vec / edge_len
    perp_left  = np.array([-edge_unit[1],  edge_unit[0]])
    perp_right = np.array([ edge_unit[1], -edge_unit[0]])

    left_whites = right_whites = 0
    for i in range(sample_count):
        t = (i + 0.5) / sample_count
        mid = c1 + t * edge_vec
        for perp, is_left in [(perp_left, True), (perp_right, False)]:
            px = mid + perp * side_offset
            xi, yi = int(round(px[0])), int(round(px[1]))
            if 0 <= xi < w and 0 <= yi < h and binary[yi, xi] > 127:
                if is_left:
                    left_whites += 1
                else:
                    right_whites += 1

    left_ratio  = left_whites  / sample_count
    right_ratio = right_whites / sample_count
    bg_thresh   = 1.0 - white_ratio_thresh

    if left_ratio >= white_ratio_thresh and right_ratio < bg_thresh:
        return "EXPOSED", perp_left
    if right_ratio >= white_ratio_thresh and left_ratio < bg_thresh:
        return "EXPOSED", perp_right
    return "INTERNAL", None


def filter_square_vertices(binary: np.ndarray,
                           all_corners: list,
                           angle_thresh: float = 25) -> list:
    """
    筛选可能是正方形顶点的角点：寻找存在两条 EXPOSED 边且夹角≈90°的角点。

    参数:
        binary: 二值图像
        all_corners: 所有角点 [(x,y), ...]
        angle_thresh: 直角容差 (度)，80°-100°算直角

    返回:
        list[np.ndarray]: 符合条件的角点
    """
    all_corners = [np.array(c, dtype=float) for c in all_corners]
    valid_corners = []

    for i, corner in enumerate(all_corners):
        exposed_edges = []
        for j, other in enumerate(all_corners):
            if i == j:
                continue
            edge_type, _ = check_edge_type(binary, corner, other)
            if edge_type == "EXPOSED":
                edge_vec = other - corner
                edge_len = np.linalg.norm(edge_vec)
                if edge_len > 1e-6:
                    exposed_edges.append(edge_vec)

        if len(exposed_edges) >= 2:
            for m in range(len(exposed_edges)):
                for n in range(m + 1, len(exposed_edges)):
                    v1 = exposed_edges[m]
                    v2 = exposed_edges[n]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    if abs(angle_deg - 90) <= angle_thresh:
                        valid_corners.append(corner)
                        break
                else:
                    continue
                break

    return valid_corners


def build_square_from_edge(c1, c2, inward_normal: np.ndarray) -> np.ndarray:
    """
    由暴露边 c1->c2 和朝内法向量构建正方形四顶点（逆时针: c1, c2, c3, c4）。

    参数:
        c1, c2: 边的两个端点
        inward_normal: 指向物体内部的单位向量

    返回:
        np.ndarray: (4, 2) float32 正方形顶点
    """
    c1     = np.array(c1, dtype=float)
    c2     = np.array(c2, dtype=float)
    side   = np.linalg.norm(c2 - c1)
    inward = inward_normal / np.linalg.norm(inward_normal) * side
    c3 = c2 + inward
    c4 = c1 + inward
    return np.array([c1, c2, c3, c4], dtype=np.float32)


def validate_square_by_area_sampling(binary: np.ndarray,
                                     square_pts: np.ndarray,
                                     n_samples: int = 50,
                                     white_thresh: float = 0.9) -> bool:
    """
    在候选正方形内部随机采样，判断是否是实心前景。

    腐蚀约 5% 边长，避免边缘噪声干扰。

    参数:
        binary: 二值图像
        square_pts: (4, 2) 正方形顶点
        n_samples: 采样点数
        white_thresh: 白色占比阈值

    返回:
        bool: True 表示确认为实心正方形
    """
    h, w = binary.shape[:2]
    pts_int = square_pts.astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int], 255)

    side       = np.linalg.norm(square_pts[1] - square_pts[0])
    erode_size = max(3, int(side * 0.05))
    kernel     = np.ones((erode_size, erode_size), np.uint8)
    eroded     = cv2.erode(mask, kernel, iterations=1)

    ys, xs = np.where(eroded > 0)
    if len(xs) == 0:
        return False

    idx         = np.random.choice(len(xs), min(n_samples, len(xs)), replace=False)
    white_count = np.sum(binary[ys[idx], xs[idx]] > 127)
    return (white_count / len(idx)) >= white_thresh


def are_squares_similar(sq1: np.ndarray,
                        sq2: np.ndarray,
                        center_dist_thresh: float = 15,
                        area_ratio_thresh: float = 0.15) -> bool:
    """中心距离近且面积相近 → 视为重复检测。"""
    def stats(sq):
        cx, cy = sq[:, 0].mean(), sq[:, 1].mean()
        area   = cv2.contourArea(sq.astype(np.float32))
        return np.array([cx, cy]), area

    c1, a1     = stats(sq1)
    c2, a2     = stats(sq2)
    dist       = np.linalg.norm(c1 - c2)
    area_ratio = abs(a1 - a2) / max(a1, a2, 1e-6)
    return dist < center_dist_thresh and area_ratio < area_ratio_thresh


def find_smallest_square(binary: np.ndarray,
                         valid_corners: list) -> np.ndarray | None:
    """
    从 valid_corners 两两配对，寻找面积最小的真实正方形。

    参数:
        binary: 二值图像
        valid_corners: 候选角点列表

    返回:
        np.ndarray | None: (4, 2) float32 最小正方形顶点，找不到返回 None
    """
    final_squares = []

    for c1, c2 in combinations(valid_corners, 2):
        edge_type, inward_v = check_edge_type(binary, c1, c2)
        if edge_type != "EXPOSED":
            continue

        candidate = build_square_from_edge(c1, c2, inward_v)

        if not validate_square_by_area_sampling(binary, candidate):
            continue

        if not any(are_squares_similar(candidate, sq) for sq in final_squares):
            final_squares.append(candidate)

    if not final_squares:
        return None

    areas = [cv2.contourArea(sq.astype(np.float32)) for sq in final_squares]
    return final_squares[int(np.argmin(areas))]
