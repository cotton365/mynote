#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serial.py — 统一串口 + 键盘推理脚本
=====================================
- 同时加载数字模型和豆子模型（两个 ONNX）
- 串口接收触发（Nano 场景）或键盘触发（PC 调试）
- 推理后串口逐类回传结果
- LIVE_PREVIEW 控制是否显示实时画面

【串口协议】
  接收（单片机 → Nano）:
    AA 00 FF  →  豆子模式（3类, N_INFER=3）
    AA 01 FF  →  数字模式（5类, N_INFER=5）

  发送（Nano → 单片机），每类 4 字节:
    AA XX 55 XX    XX = 类别编码 (01-08)

【键盘控制（PC）】
  t → 数字模式推理（仅显示，不发串口）
  y → 豆子模式推理（仅显示，不发串口）
  q → 退出

用法:
  python serial.py
"""

import os
import sys
import time
import random
import threading
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# 串口仅在 Linux 上可用（Nano），PC 上 import 失败则跳过
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[警告] pyserial 未安装，串口功能禁用")


# ===================== 显示模式 =====================
LIVE_PREVIEW = 0        # 1 = 实时预览摄像头画面
                        # 0 = 黑屏初始，按 t/y 推理后显示结果

# ===================== 相机配置 =====================
CAMERA_INDEX = 0        # /dev/video0
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080

# ===================== 模型路径（用户自己填）=====================
NUMBER_MODEL_PATH = "number2.onnx"      # 数字识别模型
BEAN_MODEL_PATH   = "best_bean.onnx"        # 豆子识别模型

# ===================== 数字模式（31.py 逻辑）=====================
NUMBER_CLASS_NAMES = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
                      5: 'c1', 6: 'c2', 7: 'c3', 8: 'c4', 9: 'c5'}
NUMBER_N_INFER = 4                              # 推理次数 → 多数投票

# 基础数字映射（c1 → '1'），投票和串口通信用
NUMBER_BASE_MAP = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
                   5: '1', 6: '2', 7: '3', 8: '4', 9: '5'}

NUMBER_CLASS_TO_CODE = {                        # 类别名 → 串口编码
    '1': 0x01, '2': 0x02, '3': 0x03,
    '4': 0x04, '5': 0x05,
}

# ===================== 豆子模式（32.py 逻辑）=====================
BEAN_CLASS_NAMES = {0: 'whitebean', 1: 'soybean', 2: 'greenbean'}
BEAN_N_INFER = 3                                # 推理次数 → 多数投票

BEAN_CLASS_TO_CODE = {                          # 类别名 → 串口编码
    'soybean':   0x06,
    'greenbean': 0x07,
    'whitebean': 0x08,
}

# ===================== 串口配置（Nano 硬件，不可动）=====================
SERIAL_PORT   = '/dev/ttyTHS1'
SERIAL_BAUD   = 115200
SERIAL_TIMEOUT = 0.3

# ===================== 串口协议常量 =====================
HEAD           = 0xAA
END            = 0x55
TRIGGER_MARKER = 0x55          # 接收指令第三字节
BEAN_MODE_ID   = 0x00          # AA 00 FF → 豆子
NUM_MODE_ID    = 0x01          # AA 01 FF → 数字

# ===================== 串口调试 =====================
SERIAL_BUFFER_DUMP_ENABLE  = True    # True = 定时打印接收缓存区
SERIAL_BUFFER_DUMP_INTERVAL = 1.0     # 秒
SERIAL_BUFFER_DUMP_LIMIT    = 64      # 每次最多打印多少字节

# ===================== 推理通用参数 =====================
IMGSZ       = 1280
CONF_THRESH = 0.001
INFER_DEVICE = 'cpu'

# ===================== 全局状态（串口线程 ↔ 主循环）=====================
trigger_lock = threading.Lock()
serial_trigger = None       # None / ('bean', 'serial') / ('number', 'serial')
running = True


# ===================== 工具函数 =====================

def draw_crop_line(frame: np.ndarray):
    """画面中间红色虚线 + 上半灰色遮罩"""
    h, w = frame.shape[:2]
    mid = h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, mid), (80, 80, 80), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for x in range(0, w, 40):
        cv2.line(frame, (x, mid), (x + 20, mid), (0, 0, 255), 2)
    cv2.putText(frame, "CROP LINE", (10, mid - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def calc_iou(box1, box2):
    """计算两个边界框的 IoU (交并比)

    Args:
        box1, box2: (x1, y1, x2, y2)

    Returns:
        float: IoU ∈ [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def has_overlap(bboxes):
    """检查 bbox 列表中是否存在任意一对重叠的框

    Args:
        bboxes: [(x1, y1, x2, y2), ...]

    Returns:
        bool: True 表示存在重叠（任意一对 IoU > 0）
    """
    n = len(bboxes)
    if n <= 1:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if calc_iou(bboxes[i], bboxes[j]) > 0:
                return True
    return False


def process_frame(frame: np.ndarray, model: YOLO, class_names: dict, base_map: dict = None,
                  sort_reverse: bool = False):
    """推理 + per-class 去重，返回 (标注图像, 类别序列)

    Args:
        frame: 已裁剪的下半部分（540p）
        model: YOLO ONNX 模型
        class_names: {cls_id: name, ...}
        base_map:    {cls_id: base_name, ...} 可选，用于 c1→'1' 合并去重
        sort_reverse: False=左→右, True=右→左
    """
    results = model(frame, imgsz=IMGSZ, conf=CONF_THRESH,
                    verbose=False, device=INFER_DEVICE)
    result = results[0]
    annotated = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(annotated, "No detection", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated, [], []

    # per-base-digit 取最高 conf（c1 和 1 合并为同一基础数字）
    best_per_base = {}  # base_key → (raw_cls_id, x1, y1, x2, y2, conf)
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        if cls_id not in class_names:
            continue
        conf = float(boxes.conf[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        base_key = cls_id % 5 if base_map else cls_id
        if base_key not in best_per_base or conf > best_per_base[base_key][5]:
            best_per_base[base_key] = (cls_id, x1, y1, x2, y2, conf)

    if not best_per_base:
        cv2.putText(annotated, "No target found", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated, [], []

    # 排序（sort_reverse=False: 左→右, True: 右→左）
    sorted_dets = sorted(best_per_base.items(), key=lambda kv: kv[1][1], reverse=sort_reverse)

    for idx, (base_key, (raw_cls, x1, y1, x2, y2, conf)) in enumerate(sorted_dets):
        color = (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{idx + 1} {class_names[raw_cls]} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"  #{idx + 1} {class_names[raw_cls]}  conf={conf:.2f}  x={x1}")

    # 底部汇总：显示用实际类名，序列用基础名称（投票/串口用）
    disp_seq = [class_names[raw_cls] for _, (raw_cls, *_) in sorted_dets]
    if base_map:
        seq = [base_map[raw_cls] for _, (raw_cls, *_) in sorted_dets]
    else:
        seq = disp_seq
    summary = " | ".join(
        [f"#{i + 1} {disp_seq[i]}" for i in range(len(disp_seq))]
    )
    cv2.putText(annotated, summary, (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    bboxes = [(x1, y1, x2, y2) for _, (_, x1, y1, x2, y2, _) in sorted_dets]
    return annotated, seq, bboxes


def do_inference(cap, model, class_names, n_infer, base_map=None, sort_reverse=False):
    """执行一次完整推理流程

    返回: (annotated_image, sequence_list)
      sequence_list: 基础数字序列 ['1','2','3','4','5'] 或 ['whitebean',...]
    """
    # 跳过缓冲帧（摄像头空闲时缓冲区积压的旧帧）
    for _ in range(4):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        return None, []

    mid = frame.shape[0] // 2
    clean_results = []
    attempt = 0
    while len(clean_results) < n_infer:
        attempt += 1
        ret, f = cap.read()
        if not ret:
            continue
        f_cropped = f[mid:, :, :]
        img, seq, bboxes = process_frame(f_cropped, model, class_names, base_map, sort_reverse)
        seq_str = "".join(seq) if seq else "无检测"
        if has_overlap(bboxes):
            print(f"  [{attempt}] {seq_str} [重叠!] 丢弃，重试...")
            continue
        clean_results.append((img, seq, bboxes))
        print(f"  [{attempt}] {seq_str} ✓  (已收集 {len(clean_results)}/{n_infer})")

    # 多数投票（全部 n_infer 个帧均无框重叠）
    seqs = [tuple(s) for _, s, _ in clean_results if s]
    if seqs:
        counter = Counter(seqs)
        most_common = counter.most_common()
        top_count = most_common[0][1]
        tied = [list(s) for s, c in most_common if c == top_count]
        winner_seq = random.choice(tied)
        # 取对应图
        display = None
        for img, s, _ in clean_results:
            if list(s) == winner_seq:
                display = img
                break
        if display is None:
            display = clean_results[0][0]
        tie_info = f" (平局 {len(tied)} 选 1)" if len(tied) > 1 else ""
        seq_str = "".join(winner_seq)
        print(f"  => 投票结果: {seq_str} ({top_count}/{n_infer}){tie_info}")
    else:
        display = clean_results[0][0] if clean_results else np.zeros_like(frame)
        winner_seq = []
        print(f"  => 全部无检测")

    return display, winner_seq


# ===================== 串口 =====================

def serial_receive_thread(ser):
    """串口接收线程：解析 AA XX FF 指令，设置 serial_trigger"""
    global running, serial_trigger
    recv_buffer = bytearray()
    last_dump_time = time.time()

    def dump_recv_buffer(prefix: str = ""):
        if not SERIAL_BUFFER_DUMP_ENABLE or not recv_buffer:
            return
        preview = bytes(recv_buffer[:SERIAL_BUFFER_DUMP_LIMIT]).hex().upper()
        suffix = " ..." if len(recv_buffer) > SERIAL_BUFFER_DUMP_LIMIT else ""
        print(f"[串口][缓存]{prefix} {len(recv_buffer)}B: {preview}{suffix}")

    while running:
        try:
            data = ser.read(10)
            if not data:
                now = time.time()
                if now - last_dump_time >= SERIAL_BUFFER_DUMP_INTERVAL:
                    dump_recv_buffer("(空闲)")
                    last_dump_time = now
                continue
            recv_buffer.extend(data)
            print(f"[串口] 收到 {len(data)}B: {data.hex().upper()}")

            while len(recv_buffer) >= 3:
                head_idx = recv_buffer.find(bytes([HEAD]))
                if head_idx == -1:
                    recv_buffer.clear()
                    break
                if len(recv_buffer) - head_idx < 3:
                    break

                b0 = recv_buffer[head_idx]
                b1 = recv_buffer[head_idx + 1]
                b2 = recv_buffer[head_idx + 2]

                if b0 == HEAD and b2 == TRIGGER_MARKER:
                    # 有效指令
                    if b1 == BEAN_MODE_ID:
                        with trigger_lock:
                            serial_trigger = ('bean', 'serial')
                        print(f"[串口] 收到豆子触发: {bytes([b0,b1,b2]).hex().upper()}")
                    elif b1 == NUM_MODE_ID:
                        with trigger_lock:
                            serial_trigger = ('number', 'serial')
                        print(f"[串口] 收到数字触发: {bytes([b0,b1,b2]).hex().upper()}")
                    else:
                        print(f"[串口] 未知模式: {bytes([b0,b1,b2]).hex().upper()}")
                    del recv_buffer[:head_idx + 3]
                    # 跳过紧随的 FF 字节（MCU 收到结果后的确认回复）
                    while len(recv_buffer) > 0 and recv_buffer[0] == 0xFF:
                        del recv_buffer[0]
                else:
                    del recv_buffer[:head_idx + 1]

            now = time.time()
            if now - last_dump_time >= SERIAL_BUFFER_DUMP_INTERVAL:
                dump_recv_buffer()
                last_dump_time = now

        except Exception as e:
            print(f"[串口] 接收异常: {e}")
            continue


def send_all(ser, seq, class_to_code):
    """一次性发送全部结果: AA XX 55 XX AA YY 55 YY ..."""
    packet = bytes()
    for name in seq:
        code = class_to_code.get(name, 0x00)
        if code == 0x00:
            print(f"[串口] 未知类别 '{name}'，跳过")
            continue
        packet += bytes([HEAD, code, END, code])
    if packet:
        try:
            ser.write(packet)
            print(f"[串口] 发送: {' '.join(seq)} → {packet.hex().upper()}")
        except Exception as e:
            print(f"[串口] 发送失败: {e}")


def open_serial():
    """打开串口，启动接收线程；失败返回 None"""
    if not HAS_SERIAL:
        print("[串口] pyserial 不可用，跳过")
        return None
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        print(f"[串口] {SERIAL_PORT} 打开成功 (baud={SERIAL_BAUD})")
        t = threading.Thread(target=serial_receive_thread, args=(ser,), daemon=True)
        t.start()
        return ser
    except Exception as e:
        print(f"[串口] 打开失败: {e}（继续运行，仅键盘可用）")
        return None


# ===================== 主函数 =====================

def main():
    global running, serial_trigger

    # ---- 加载模型 ----
    print(f"加载数字模型: {NUMBER_MODEL_PATH}")
    num_model = YOLO(NUMBER_MODEL_PATH, task='detect')

    print(f"加载豆子模型: {BEAN_MODEL_PATH}")
    bean_model = YOLO(BEAN_MODEL_PATH, task='detect')

    # ---- 打开摄像头 ----
    print(f"打开摄像头: /dev/video{CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"无法打开摄像头 {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
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

    for _ in range(5):
        cap.read()

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"分辨率: {int(actual_w)}×{int(actual_h)}")

    # ---- 打开串口 ----
    ser = open_serial()

    # ---- 初始化显示 ----
    last_result = None

    if LIVE_PREVIEW:
        print("\n按键:  t=数字推理  y=豆子推理  q=退出\n")
        black_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    else:
        print("\n无界面模式，仅串口触发 (Ctrl+C 退出)\n")

    # ---- 主循环 ----
    while running:
        if LIVE_PREVIEW:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

            # ── 确定本次触发 ──
            trigger = None  # (mode, source)

            # 键盘触发
            if key == ord('t'):
                trigger = ('number', 'keyboard')
            elif key == ord('y'):
                trigger = ('bean', 'keyboard')

            # 串口触发（优先于键盘）
            with trigger_lock:
                if serial_trigger is not None:
                    trigger = serial_trigger
                    serial_trigger = None

            # ── 显示画面 ──
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()

        else:
            # 无界面模式：仅串口触发，轮询等待
            time.sleep(0.05)

            with trigger_lock:
                if serial_trigger is not None:
                    trigger = serial_trigger
                    serial_trigger = None
                else:
                    trigger = None

        # ── 执行推理 ──
        if trigger is not None:
            mode, source = trigger
            source_tag = "键盘" if source == 'keyboard' else "串口"

            # 串口触发：推理前清空接收缓冲，丢弃旧数据
            if source == 'serial' and ser:
                ser.reset_input_buffer()

            if mode == 'number':
                print(f"\n[{source_tag}] 触发数字模式 (N_INFER={NUMBER_N_INFER})")
                annotated, seq = do_inference(cap, num_model,
                                              NUMBER_CLASS_NAMES, NUMBER_N_INFER,
                                              NUMBER_BASE_MAP)

                # 串口回传（仅串口触发）
                if source == 'serial' and ser and seq:
                    send_all(ser, seq, NUMBER_CLASS_TO_CODE)

            else:  # bean
                print(f"\n[{source_tag}] 触发豆子模式 (N_INFER={BEAN_N_INFER})")
                annotated, seq = do_inference(cap, bean_model,
                                              BEAN_CLASS_NAMES, BEAN_N_INFER,
                                              sort_reverse=True)

                # 串口回传（仅串口触发）
                if source == 'serial' and ser and seq:
                    send_all(ser, seq, BEAN_CLASS_TO_CODE)

            if annotated is not None:
                last_result = annotated
                display = annotated
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         f"result_{ts}.jpg")
                cv2.imwrite(save_path, annotated)
                print(f"已保存: {save_path}")

                if seq:
                    seq_str = "  ".join(seq)
                    print(f"识别结果: {seq_str}")
                print()

            # 串口触发：推理完成后清空接收缓冲，丢弃期间积压的重复 AA XX FF
            if source == 'serial' and ser:
                ser.reset_input_buffer()

        # ── 渲染显示 ──
        if LIVE_PREVIEW:
            display = cv2.resize(display, (CAMERA_WIDTH, CAMERA_HEIGHT),
                                 interpolation=cv2.INTER_LINEAR)
            hint = "T=Number  Y=Bean  Q=Quit"
            cv2.putText(display, hint, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if trigger is None:
                draw_crop_line(display)
            cv2.imshow("Serial - ONNX Inference", display)

    # ---- 清理 ----
    running = False
    cap.release()
    if ser:
        ser.close()
    if LIVE_PREVIEW:
        cv2.destroyAllWindows()
    print("退出")


if __name__ == "__main__":
    main()
