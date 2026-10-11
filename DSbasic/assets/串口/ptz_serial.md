"""
云台串口通信模块 (硬件版)
=====================

上位机 (Raspberry Pi 4B) → 下位机 (MSPM0G350X) 的 UART 通信封装。
负责帧格式化、收发控制、FPS/频率统计。

类:
    PTZSerial
        封装偏差计算、帧发送 (正常/异常)、FPS统计。
        协议: 正常帧 [±DDD±DDD*] (11字节), 异常帧 (x=999,y=999) (13字节)
        UART: 115200 bps, 8N1

用法:
    ptz = PTZSerial(port="/dev/serial0", baudrate=115200, image_width=480, image_height=320)
    ptz.open()
    ptz.send_deviation(cx, cy)  # 发送像素偏差
    ptz.send_no_target()        # 发送无目标帧
    ptz.close()

参数:
    port: 串口设备路径，默认 "/dev/serial0"
    baudrate: 波特率，默认 115200
    image_width: 摄像头图像宽度 (px)
    image_height: 摄像头图像高度 (px)

依赖:
    pyserial (serial)
"""

import time
import serial


class PTZSerial:
    """云台串口通信，封装帧格式化与收发，附带 FPS/发送频率统计"""

    def __init__(self, port="/dev/serial0", baudrate=115200,
                 image_width=480, image_height=320):
        """
        参数:
            port: 串口设备路径
            baudrate: 波特率
            image_width: 摄像头图像宽度 (px)
            image_height: 摄像头图像高度 (px)
        """
        self.port = port
        self.baudrate = baudrate
        self.center_x = image_width / 2
        self.center_y = image_height / 2
        self._ser = None

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
        """打开串口"""
        self._ser = serial.Serial(
            self.port,
            self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.01,
        )

    def close(self):
        """关闭串口"""
        if self._ser is not None and self._ser.is_open:
            self._ser.close()

    @property
    def is_open(self):
        return self._ser is not None and self._ser.is_open

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
        if self._ser is None or not self._ser.is_open:
            return 0
        data = b"(x=999,y=999)"
        self._send_count += 1
        self._no_target_count += 1
        return self._ser.write(data)

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
        if self._ser is None or not self._ser.is_open:
            return 0
        frame = f"[{wx:+04d}{wy:+04d}*]"
        self._send_count += 1
        return self._ser.write(frame.encode())
