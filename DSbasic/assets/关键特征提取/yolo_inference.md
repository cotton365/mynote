"""
YOLO ONNX 推理管线模块
===================

基于 Ultralytics YOLO 模型的单帧/多帧推理 + per-class 去重 + 多数投票。

函数:
    calc_iou(box1, box2) -> float
        计算两个边界框的 IoU (交并比)
    has_overlap(bboxes) -> bool
        检查 bbox 列表中是否存在任意重叠框对
    process_frame(frame, model, class_names, base_map, sort_reverse) -> tuple
        单帧推理 + per-base-class 去重（取 confidence 最高的框）
    do_inference(cap, model, class_names, n_infer, base_map, sort_reverse) -> tuple
        多帧推理 + 重叠丢弃 + 多数投票

参数:
    各函数参数见具体 docstring。

来源:
    ser.py — 数字识别 & 豆子分类的 YOLO 推理核心逻辑
"""

import random
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


def calc_iou(box1: tuple, box2: tuple) -> float:
    """
    计算两个边界框的 IoU (交并比)。

    参数:
        box1, box2: (x1, y1, x2, y2)

    返回:
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


def has_overlap(bboxes: list) -> bool:
    """
    检查 bbox 列表中是否存在任意一对重叠的框。

    参数:
        bboxes: [(x1, y1, x2, y2), ...]

    返回:
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


def process_frame(frame: np.ndarray,
                  model: YOLO,
                  class_names: dict,
                  base_map: dict = None,
                  sort_reverse: bool = False,
                  imgsz: int = 1280,
                  conf_thresh: float = 0.001,
                  device: str = 'cpu') -> tuple:
    """
    单帧推理 + per-class 去重，返回标注图像和类别序列。

    参数:
        frame: BGR 图像（通常为裁剪后的下半部分 540p）
        model: YOLO ONNX 模型
        class_names: {cls_id: name, ...}
        base_map: {cls_id: base_name, ...} 可选，用于 c1→'1' 合并去重
        sort_reverse: False=左→右排序, True=右→左排序
        imgsz: 推理尺寸
        conf_thresh: 置信度阈值
        device: 推理设备

    返回:
        (annotated: np.ndarray, seq: list[str], bboxes: list[tuple])
        annotated: 带检测框和编号的 BGR 图像副本
        seq: 基础类别序列
        bboxes: [(x1, y1, x2, y2), ...]
    """
    results = model(frame, imgsz=imgsz, conf=conf_thresh,
                    verbose=False, device=device)
    result = results[0]
    annotated = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(annotated, "No detection", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated, [], []

    # per-base-digit 取最高 conf
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

    # 排序
    sorted_dets = sorted(best_per_base.items(), key=lambda kv: kv[1][1], reverse=sort_reverse)

    for idx, (base_key, (raw_cls, x1, y1, x2, y2, conf)) in enumerate(sorted_dets):
        color = (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{idx + 1} {class_names[raw_cls]} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 底部汇总
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


def do_inference(cap: cv2.VideoCapture,
                 model: YOLO,
                 class_names: dict,
                 n_infer: int,
                 base_map: dict = None,
                 sort_reverse: bool = False,
                 imgsz: int = 1280,
                 conf_thresh: float = 0.001,
                 device: str = 'cpu') -> tuple:
    """
    多帧推理 + 重叠丢弃 + 多数投票。

    流程:
        1. 跳过 4 帧缓冲
        2. 读取 n_infer 帧有效帧（丢弃有重叠框的帧）
        3. 对序列做多数投票，平局随机选一
        4. 返回投票优胜帧的标注图

    参数:
        cap: 已打开的 cv2.VideoCapture
        model: YOLO ONNX 模型
        class_names: {cls_id: name, ...}
        n_infer: 所需有效推理帧数
        base_map: {cls_id: base_name, ...} 可选
        sort_reverse: 排序方向
        imgsz: 推理尺寸
        conf_thresh: 置信度阈值
        device: 推理设备

    返回:
        (display: np.ndarray | None, winner_seq: list[str])
        display: 优胜帧的标注图像，失败返回 None
        winner_seq: 最终类别序列
    """
    # 跳过缓冲帧
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
        img, seq, bboxes = process_frame(f_cropped, model, class_names,
                                         base_map, sort_reverse,
                                         imgsz=imgsz, conf_thresh=conf_thresh,
                                         device=device)
        seq_str = "".join(seq) if seq else "无检测"
        if has_overlap(bboxes):
            continue
        clean_results.append((img, seq, bboxes))

    # 多数投票
    seqs = [tuple(s) for _, s, _ in clean_results if s]
    if seqs:
        counter = Counter(seqs)
        most_common = counter.most_common()
        top_count = most_common[0][1]
        tied = [list(s) for s, c in most_common if c == top_count]
        winner_seq = random.choice(tied)
        display = None
        for img, s, _ in clean_results:
            if list(s) == winner_seq:
                display = img
                break
        if display is None:
            display = clean_results[0][0]
    else:
        display = clean_results[0][0] if clean_results else np.zeros_like(frame)
        winner_seq = []

    return display, winner_seq
