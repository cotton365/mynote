"""
云台串口通信模块 (PC 调试版)
=======================

PC 端无硬件串口的 PTZSerial 模拟实现。
接口与硬件版完全一致，所有发送操作改为控制台打印。

类:
    PTZSerial
        Mock 版云台串口 — 打印帧内容到控制台，FPS 统计保留。

用法:
    ptz = PTZSerial(port="/dev/serial0", baudrate=115200, image_width=480, image_height=320)
    ptz.open()
    ptz.send_deviation(cx, cy)  # 打印到控制台
    ptz.send_no_target()
    ptz.close()

参数:
    port: 忽略
    baudrate: 忽略
    image_width: 摄像头图像宽度 (px)
    image_height: 摄像头图像高度 (px)

依赖:
    无外部依赖（仅标准库 time）
"""

import time


class PTZSerial:
    """云台串口模拟 — PC 调试用，帧格式化 + FPS 统计，不真正发串口"""

    def __init__(self, port="/dev/serial0", baudrate=115200,
                 image_width=480, image_height=320):
        """
        参数:
            port: 串口设备路径 (忽略)
            baudrate: 波特率 (忽略)
            image_width: 摄像头图像宽度 (px)
            image_height: 摄像头图像高度 (px)
        """
        self.port = port
        self.baudrate = baudrate
        self.center_x = image_width / 2
        self.center_y = image_height / 2

        # ---- 打印节流：每 0.5s 最多打印一次 ----
        self._last_print_time = 0.0
        self._print_interval = 0.5

        # ---- FPS / 发送频率统计 ----
        self._frame_count = 0
        self._send_count = 0
        self._no_target_count = 0
        self._last_stats_time = time.time()
        self._report_interval = 2.0
        self.main_fps = 0.0
        self.send_hz = 0.0
        self.no_target_hz = 0.0

    # ---- 生命周期 ----

    def open(self):
        """打开串口 (PC 版无操作)"""
        print(f"[Serial] Mock opened (port={self.port}, baud={self.baudrate})")

    def close(self):
        """关闭串口 (PC 版无操作)"""
        print("[Serial] Mock closed")

    @property
    def is_open(self):
        return True   # 始终返回 True

    # ---- 偏差计算 ----

    def calc_deviation(self, cx, cy):
        """
        计算矩形中心相对于画面中心的像素偏差。

        协议方向:
          wucha_x = center_x - cx  (正 = 矩形偏左)
          wucha_y = center_y - cy  (正 = 矩形偏上)

        返回: (wucha_x: float, wucha_y: float)
        """
        return (self.center_x - cx, self.center_y - cy)

    # ---- 帧发送 ----

    def send_deviation(self, cx, cy):
        """
        发送正常偏差帧: [±DDD±DDD*]

        参数:
            cx, cy: 矩形中心在画面中的坐标 (px)
        """
        wx, wy = self.calc_deviation(cx, cy)
        return self._send_raw(int(round(wx)), int(round(wy)))

    def send_raw(self, wucha_x, wucha_y):
        """
        直接发送已算好的偏差值 (限幅 ±999)。

        参数:
            wucha_x, wucha_y: 像素偏差值
        """
        wx = max(-999, min(999, int(round(wucha_x))))
        wy = max(-999, min(999, int(round(wucha_y))))
        return self._send_raw(wx, wy)

    def send_no_target(self):
        """发送无目标异常帧: (x=999,y=999)"""
        self._send_count += 1
        self._no_target_count += 1
        self._throttled_print("(x=999,y=999)")
        return 13

    # ---- FPS / 频率统计 ----

    def tick(self):
        """
        每个处理帧调用一次，更新 FPS 统计。

        返回: 当统计刷新时返回 (main_fps, send_hz, no_target_hz)，否则返回 None
        """
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_stats_time

        if elapsed >= self._report_interval:
            self.main_fps = self._frame_count / elapsed
            self.send_hz = self._send_count / elapsed
            self.no_target_hz = self._no_target_count / elapsed

            self._frame_count = 0
            self._send_count = 0
            self._no_target_count = 0
            self._last_stats_time = now

            return (self.main_fps, self.send_hz, self.no_target_hz)

        return None

    # ---- 内部 ----

    def _send_raw(self, wx, wy):
        """发送 [+DDD-DDD*] 帧，wx/wy 已限幅为 int"""
        self._send_count += 1
        frame = f"[{wx:+04d}{wy:+04d}*]"
        self._throttled_print(frame)
        return len(frame)

    def _throttled_print(self, msg):
        """节流打印，避免刷屏"""
        now = time.time()
        if now - self._last_print_time >= self._print_interval:
            print(f"[Serial] {msg}")
            self._last_print_time = now
