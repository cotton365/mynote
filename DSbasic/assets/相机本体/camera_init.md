"""
相机初始化模块
===========

打开 USB/工业相机，设置 MJPG 格式、分辨率、帧率和全部图像参数。
预热数帧后返回可用的 cv2.VideoCapture 对象。

函数:
    init_camera(index=0, width=1920, height=1080, fps=30) -> cv2.VideoCapture
        打开并配置相机，返回已预热的 VideoCapture 实例。
        失败时抛出 RuntimeError。

参数:
    index : int
        相机设备索引，默认 0 (/dev/video0)
    width : int
        采集宽度 (px)，默认 1920
    height : int
        采集高度 (px)，默认 1080
    fps : int
        采集帧率，默认 30
"""

import cv2


def init_camera(index: int = 0, width: int = 1920, height: int = 1080,
                fps: int = 30) -> cv2.VideoCapture:
    """
    打开相机，设置全部参数，预热 5 帧后返回。

    返回:
        cv2.VideoCapture: 已打开并配置好的相机实例

    异常:
        RuntimeError: 相机无法打开
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("[Camera] 无法打开相机")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    cap.set(cv2.CAP_PROP_SATURATION, 64)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_SHARPNESS, 10)
    cap.set(cv2.CAP_PROP_GAMMA, 100)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(5):
        cap.read()

    print(f"[Camera] 相机已初始化: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
          f"{int(cap.get(cv2.CAP_PROP_FPS))}fps")
    return cap
