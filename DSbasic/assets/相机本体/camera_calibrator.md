"""
棋盘格相机内参标定模块
===============

使用棋盘格图片集计算相机内参矩阵和畸变系数，输出 OpenCV YAML 文件。

函数:
    calibrate_camera(image_dir, pattern_size, square_size, output_path) -> dict
        执行标定，返回内参矩阵、畸变系数、RMS 重投影误差。

参数:
    image_dir : str
        棋盘格图片所在目录，默认 "captures"
    pattern_size : tuple[int, int]
        棋盘格内角点数量 (cols, rows)，默认 (9, 6)
    square_size : float
        每个格子的真实尺寸 (mm)，默认 55.0
    output_path : str
        输出 YAML 路径，默认 "camera_params.yaml"

返回:
    dict: {"camera_matrix": np.ndarray (3,3),
           "dist_coeffs": np.ndarray,
           "rms_error": float}
"""

import cv2
import numpy as np
import glob
import os


def calibrate_camera(image_dir: str = "captures",
                     pattern_size: tuple = (9, 6),
                     square_size: float = 55.0,
                     output_path: str = "camera_params.yaml") -> dict:
    """
    执行棋盘格相机标定，保存结果到 YAML 文件。

    流程:
        1. glob 读取 image_dir 下所有 .png
        2. 逐张检测棋盘格角点 (cv2.findChessboardCornersSB)
        3. cv2.calibrateCamera 求解内参和畸变
        4. 保存 camera_matrix / dist_coeff 到 output_path

    参数:
        image_dir: 包含棋盘格照片的目录路径
        pattern_size: 棋盘格内角点数 (cols, rows)
        square_size: 格子边长 (mm)
        output_path: 输出 YAML 文件路径

    返回:
        {"camera_matrix": (3,3) np.ndarray,
         "dist_coeffs": np.ndarray,
         "rms_error": float}
    """
    images = glob.glob(os.path.join(image_dir, "*.png"))

    if not images:
        raise FileNotFoundError(f"在 {image_dir} 下未找到 .png 图片")

    # 世界坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    gray = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(
            gray, pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        )

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"[成功] {fname}")
        else:
            print(f"[失败] {fname}")

    if not objpoints:
        raise RuntimeError("没有成功检测到任何棋盘格角点")

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"\nRMS误差: {rms:.4f}")
    print(f"\nCamera Matrix:\n{mtx}")
    print(f"\nDistortion:\n{dist}")

    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", mtx)
    fs.write("dist_coeff", dist)
    fs.release()

    print(f"\n已保存: {output_path}")

    return {
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "rms_error": rms,
    }
