"""
帧采集模块
=======

从相机采集帧，按空格键保存到指定目录，按 q 退出。

函数:
    next_index(folder: str) -> int
        获取目录中下一个可用编号（从0开始递增）
    capture_frames_interactive(save_dir, camera_index, width, height) -> None
        打开相机，交互式采集帧（阻塞式循环）

参数:
    save_dir : str
        图片保存目录，默认 "captures"
    camera_index : int
        相机设备索引，默认 0
    width : int
        采集宽度 (px)，默认 1920
    height : int
        采集高度 (px)，默认 1080
"""

import os
import cv2


def next_index(folder: str) -> int:
    """扫描 folder 下 .jpg 文件，返回下一个可用整数编号。"""
    os.makedirs(folder, exist_ok=True)
    max_idx = -1
    for name in os.listdir(folder):
        if name.lower().endswith(".jpg"):
            stem = os.path.splitext(name)[0]
            if stem.isdigit():
                max_idx = max(max_idx, int(stem))
    return max_idx + 1


def capture_frames_interactive(save_dir: str = "captures",
                               camera_index: int = 0,
                               width: int = 1920,
                               height: int = 1080) -> None:
    """
    打开相机，进入交互式帧采集模式。

    控制:
        空格 → 保存当前帧到 save_dir/N.png
        q    → 退出

    参数:
        save_dir: 图片保存目录
        camera_index: 相机索引 (/dev/videoX)
        width: 采集宽度
        height: 采集高度
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    cap.set(cv2.CAP_PROP_SATURATION, 64)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_SHARPNESS, 10)
    cap.set(cv2.CAP_PROP_GAMMA, 100)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    idx = next_index(save_dir)
    print("按空格拍照，按 q 退出")

    while True:
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        if not ret:
            print("读取帧失败")
            break

        scale = 0.5
        show = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Capture", show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            filename = f"{idx}.png"
            path = os.path.join(save_dir, filename)
            cv2.imwrite(path, frame)
            print(f"已保存: {path}")
            print(f"Resolution: {w}x{h}")
            idx += 1

    cap.release()
    cv2.destroyAllWindows()
