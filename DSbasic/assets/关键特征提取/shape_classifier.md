"""
形状分类模块 — 基于极坐标 FFT 分析
============================

对轮廓进行极坐标半径采样，通过 FFT 频谱特征将形状分类为 circle / square / triangle / hexagon。

函数:
    polar_radius_analysis(contour, n_samples=250) -> dict
        极坐标半径分析，返回 FFT 主频、能量比、半径方差等
    classify_object(contour) -> tuple[str, float]
        综合圆度和极坐标 FFT 特征进行形状分类

参数:
    contour : np.ndarray
        OpenCV 轮廓 (N, 1, 2) 或 (N, 2)

返回:
    classify_object: (shape_type: str, size_px: float)
        shape_type: "circle" | "square" | "triangle" | "hexagon" | "unknown"
        size_px: circle→半径, square→minAreaRect短边, 其他→包围圆半径
"""

import cv2
import numpy as np


def polar_radius_analysis(contour: np.ndarray, n_samples: int = 250) -> dict:
    """
    极坐标半径分析 — 对轮廓等角度采样半径序列，做 FFT 提取频谱特征。

    参数:
        contour: OpenCV 轮廓 (N, 1, 2) 或 (N, 2)
        n_samples: 角度采样点数，默认 250

    返回:
        dict:
            dominant_freq : int — 2-9 范围内的主频
            energy_ratio  : float — 主频能量占比
            radius_std    : float — 归一化半径标准差
            radii_norm    : np.ndarray — 归一化半径序列
            square_energy : float — 频率4的能量
            hex_energy    : float — 频率6的能量
    """
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return {"dominant_freq": 0, "energy_ratio": 0, "radius_std": 0,
                "radii_norm": [], "square_energy": 0.0, "hex_energy": 0.0}

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    pts = contour.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx**2 + dy**2)

    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]

    sample_angles = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    radii_resampled = np.interp(
        sample_angles, angles_sorted, radii_sorted, period=2*np.pi
    )

    mean_r = np.mean(radii_resampled)
    radii_norm = radii_resampled / mean_r

    fft_vals = np.abs(np.fft.rfft(radii_norm - 1.0))
    square_energy = float(fft_vals[4])
    hex_energy = float(fft_vals[6])
    freqs = np.fft.rfftfreq(n_samples, d=1.0/n_samples)

    valid_mask = (freqs >= 2) & (freqs <= 9)
    fft_valid = fft_vals.copy()
    fft_valid[~valid_mask] = 0

    dominant_freq = int(round(freqs[np.argmax(fft_valid)]))
    dominant_energy = fft_valid[np.argmax(fft_valid)]
    total_energy = np.sum(fft_vals[valid_mask]) + 1e-6
    energy_ratio = dominant_energy / total_energy

    radius_std = np.std(radii_norm)

    return {
        "dominant_freq": dominant_freq,
        "energy_ratio": energy_ratio,
        "radius_std": radius_std,
        "radii_norm": radii_norm,
        "square_energy": square_energy,
        "hex_energy": hex_energy,
    }


def classify_object(contour: np.ndarray) -> tuple:
    """
    综合圆度和极坐标 FFT 特征进行形状分类。

    分类逻辑:
        1. circularity < 0.70 → triangle
        2. 高圆度区:
           - std < 0.05 且 energy_ratio < 0.5 → circle
           - 主频 4-5 且 square_energy/hex_energy > 0.7 → square
           - 主频=6 且 square_energy/hex_energy ≤ 0.7 → hexagon
           - circularity > 0.95 → circle, 否则 hexagon (兜底)

    参数:
        contour: OpenCV 轮廓 (N, 1, 2)

    返回:
        (shape_type: str, size_px: float)
        shape_type: "circle" | "square" | "triangle" | "hexagon" | "unknown"
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return ("unknown", 0.0)

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    polar = polar_radius_analysis(contour)
    freq = polar["dominant_freq"]
    ratio = polar["energy_ratio"]
    std = polar["radius_std"]
    square_energy = polar["square_energy"]
    hex_energy = polar["hex_energy"]

    if circularity < 0.70:
        shape = "triangle"
    else:
        if std < 0.05 and ratio < 0.5:
            shape = "circle"
        elif 4 <= freq <= 5 and (square_energy / max(hex_energy, 1e-6)) > 0.7:
            shape = "square"
        elif freq == 6 and (square_energy / max(hex_energy, 1e-6)) <= 0.7:
            shape = "hexagon"
        else:
            shape = "circle" if circularity > 0.95 else "hexagon"

    if shape == "circle":
        (x, y), radius = cv2.minEnclosingCircle(contour)
        size_px = float(radius)
    elif shape == "square":
        rect = cv2.minAreaRect(contour)
        size_px = float(min(rect[1]))
    else:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        size_px = float(radius)

    return (shape, size_px)
