import cv2
import math
import numpy as np
from serial2 import PTZSerial


# def show_resized(window_name, img, scale=0.5):
#     show = cv2.resize(img, (0, 0), fx=scale, fy=scale)
#     cv2.imshow(window_name, show)


# ===================== 矩形校验工具函数 =====================

def _are_parallel(theta1, theta2, tolerance=30):
    """两个方向角是否平行（容差 ±tolerance 度）"""
    diff = abs(theta1 - theta2) % 180
    if diff > 90:
        diff = 180 - diff
    return diff <= tolerance


def _are_perpendicular(theta1, theta2, tolerance=30):
    """两个方向角是否垂直（容差 ±tolerance 度）"""
    diff = abs(theta1 - theta2) % 180
    if diff > 90:
        diff = 180 - diff
    return abs(diff - 90) <= tolerance


def check_edge_type(binary, c1, c2, sample_count=20, side_offset=8, shape_ratio_thresh=0.7):
    """
    检查边 c1->c2 在二值图中的内外属性。

    二值图经过 bitwise_not，白色(255)=目标形状，黑色(0)=背景。
    沿边采样，在边的左右两侧各取 sample_count 个点：
      - 左侧（逆时针90°）白色占比 > thresh 且 右侧白色占比 < 1-thresh → EXPOSED, inward=左
      - 反之 → EXPOSED, inward=右
      - 都不满足 → INTERNAL

    返回: (edge_type, inward_normal)
      edge_type: "EXPOSED" | "INTERNAL" | "UNKNOWN"
      inward_normal: 指向形状（白色）侧的单位向量（仅 EXPOSED 有效）
    """
    h, w = binary.shape[:2]
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    edge_vec = c2 - c1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-6:
        return "UNKNOWN", None

    edge_unit = edge_vec / edge_len
    perp_left = np.array([-edge_unit[1], edge_unit[0]])   # 逆时针 90°
    perp_right = np.array([edge_unit[1], -edge_unit[0]])   # 顺时针 90°

    left_whites = right_whites = 0
    for i in range(sample_count):
        t = (i + 0.5) / sample_count
        mid = c1 + t * edge_vec
        for perp, is_left in [(perp_left, True), (perp_right, False)]:
            px = mid + perp * side_offset
            xi, yi = int(round(px[0])), int(round(px[1]))
            if 0 <= xi < w and 0 <= yi < h and binary[yi, xi] >= 127:
                if is_left:
                    left_whites += 1
                else:
                    right_whites += 1

    left_ratio = left_whites / sample_count
    right_ratio = right_whites / sample_count
    bg_thresh = 1.0 - shape_ratio_thresh

    if left_ratio >= shape_ratio_thresh and right_ratio < bg_thresh:
        return "EXPOSED", perp_left
    if right_ratio >= shape_ratio_thresh and left_ratio < bg_thresh:
        return "EXPOSED", perp_right
    return "INTERNAL", None


def perspective_center(pts):
    """
    透视矫正后的四边形中心。

    将 4 个角点视作矩形的透视投影，通过单应变换矫正为正规矩形，
    取矫正后矩形的中心，再映射回原始图像坐标。

    Args:
        pts: (4, 2) array，四边形四个角点（按轮廓顺序）

    Returns:
        (cx, cy): 透视矫正后的中心坐标，退化时回退到质心
    """
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"perspective_center expected (4,2), got {pts.shape}")

    # 目标矩形尺寸：对边平均长度
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


def is_valid_rect(pts, min_side=20, angle_tol=30, opp_side_ratio=0.6):
    """
    校验四边形是否满足几何矩形约束（参考 矩形8.3.1_矩形识别约束分析.md）。

    约束:
      1. 恰好 4 个顶点
      2. 每条边 > min_side px
      3. 对边平行（容差 ±angle_tol°）
      4. 邻边垂直（容差 ±angle_tol°）
      5. 对边长度比 > opp_side_ratio

    返回: (is_valid: bool, detail: str)
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


def is_valid_frame(pts, binary, min_side=20, angle_tol=30, opp_side_ratio=0.6,
                   side_offset=8, shape_ratio_thresh=0.7):
    """
    校验四边形是否为黑框边缘（内框或外框都接受）。

    分两层校验:
      A. 几何矩形约束（is_valid_rect）
      B. 黑框边属性：每条边必须满足一侧是形状(白)、一侧是背景(黑)（EXPOSED），
         且 4 条边的形状侧朝向必须一致：
           - 全部形状侧朝外 → "outer"（外框）
           - 全部形状侧朝内 → "inner"（内框）
           - 不一致 → 拒绝

    返回: (is_valid: bool, detail: str, frame_type: str)
      frame_type: "outer" | "inner" | ""（无效时为空）
    """
    # ---- A. 几何约束 ----
    ok, detail = is_valid_rect(pts, min_side, angle_tol, opp_side_ratio)
    if not ok:
        return False, detail, ""

    # ---- B. 黑框边属性：一侧形状、一侧背景，且方向一致 ----
    center = np.mean(pts, axis=0)
    inward_count = 0   # 形状侧朝内的边数

    for i in range(4):
        c1 = pts[i]
        c2 = pts[(i + 1) % 4]
        edge_type, inward_normal = check_edge_type(
            binary, c1, c2,
            side_offset=side_offset,
            shape_ratio_thresh=shape_ratio_thresh,
        )

        if edge_type != "EXPOSED":
            return False, f"边{i}非暴露边({edge_type})", ""

        # inward_normal 指向形状（白色）侧
        # dot > 0 → 形状侧朝外（外框）; dot < 0 → 形状侧朝内（内框）
        edge_mid = (np.array(c1, dtype=float) + np.array(c2, dtype=float)) / 2.0
        to_center = center - edge_mid
        if np.dot(inward_normal, to_center) < 0:
            inward_count += 1

    if inward_count == 0:
        return True, "OK (外框)", "outer"
    elif inward_count == 4:
        return True, "OK (内框)", "inner"
    else:
        return False, f"黑侧朝向不一致(inward={inward_count}/4)", ""


# ===================== 黑色形状提取 =====================

def extract_black_shape_vertices(
    bgr,
    threshold_val=80,
    min_area=500,
    epsilon_ratio=0.02,
    do_morph=True,
    rect_min_side=20,
    rect_angle_tol=30,
    rect_opp_side_ratio=0.6,
    frame_side_offset=8,
    frame_shape_ratio=0.7,
):

    vis = bgr.copy()

    # ===== 阶段1：预处理 =====
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary_display = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    binary_display = cv2.bitwise_not(binary_display)  # 反转：黑色形状变白色轮廓，便于 findContours

    if do_morph:
        kernel = np.ones((3, 3), np.uint8)
        binary_display = cv2.morphologyEx(binary_display, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_display = cv2.morphologyEx(binary_display, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ===== 阶段2：特征提取 =====
    contours, _ = cv2.findContours(binary_display, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    vertices_list = []
    best_frame = None  # (pts, area) — 只取面积最大的一个黑框

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)

        pts = approx.reshape(-1, 2)
        vertices_list.append(pts)

        # 画轮廓（绿线）
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

        # ===== 阶段3：黑框边校验 =====
        is_frame = False
        if len(pts) == 4:
            ok, detail, frame_type = is_valid_frame(
                pts, binary_display,
                min_side=rect_min_side,
                angle_tol=rect_angle_tol,
                opp_side_ratio=rect_opp_side_ratio,
                side_offset=frame_side_offset,
                shape_ratio_thresh=frame_shape_ratio,
            )
            if ok:
                # ---- 凸性检查 (from A4.py, git_aim/aim.py) ----
                hull = cv2.convexHull(approx)
                hull_area = cv2.contourArea(hull)
                if hull_area >= 1 and area / hull_area >= 0.85:
                    is_frame = True
                    if best_frame is None or area > best_frame[1]:
                        best_frame = (pts, area)

        # 画角点：帧角点=青色，非帧=红角点
        corner_color = (255, 255, 0) if is_frame else (0, 0, 255)
        for (x, y) in pts:
            cv2.circle(vis, (int(x), int(y)), 6, corner_color, -1)

    # ===== 阶段4：面积最大的黑框 vs 相机中心的偏移 =====
    frame_center = None
    if best_frame is not None:
        best_pts = best_frame[0]
        fc_x, fc_y = perspective_center(best_pts)           # 透视矫正后的黑框中心
        frame_center = (fc_x, fc_y)
        h, w = bgr.shape[:2]
        cam_x, cam_y = w / 2, h / 2                         # 相机中心

        # 画中心点（绿色=框中心，黄色=相机中心）
        cv2.circle(vis, (int(fc_x), int(fc_y)), 8, (0, 255, 0), -1)
        cv2.circle(vis, (int(cam_x), int(cam_y)), 5, (0, 255, 255), -1)

    return {
        "vis": vis,
        "gray": gray,
        "binary": binary_display,
        "vertices": vertices_list,
        "frame_center": frame_center,
    }


# ===================== 相机配置 =====================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
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
    raise RuntimeError("Cannot open camera")

# 读取一帧获取实际分辨率 (cap.set 可能不生效)
ret, test_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read from camera")
actual_h, actual_w = test_frame.shape[:2]
print(f"Actual resolution: {actual_w}x{actual_h}")

# ---- 串口初始化 (使用实际分辨率) ----
ptz = PTZSerial(port="/dev/serial0", baudrate=115200,
                image_width=actual_w, image_height=actual_h)
ptz.open()
print("Serial opened:", ptz.is_open)

threshold_val = 125
stats = None  # (main_fps, send_hz, no_target_hz)
bw_locked = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_black_shape_vertices(
        frame,
        threshold_val=threshold_val,
        min_area=600,
        epsilon_ratio=0.02,
        do_morph=True,
        rect_min_side=20,
        rect_angle_tol=14,          # 调优值 (trackbar_params.json)
        rect_opp_side_ratio=0.38,   # 调优值 (trackbar_params.json)
        frame_shape_ratio=0.0,      # 调优值 (trackbar_params.json)
    )

    # ---- 串口发送 ----
    best_bw_center = None
    if feat["frame_center"] is not None:
        fc_x, fc_y = feat["frame_center"]

        # ---- 近距亮蓝检测 (|dx|+|dy| < 20) ----
        cam_x, cam_y = actual_w / 2, actual_h / 2
        if bw_locked or abs(fc_x - cam_x) + abs(fc_y - cam_y) < 20:
            # 预处理：CLAHE + GaussianBlur（与 blue_white.py 一致）
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b])
            frame_bw = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

            hsv = cv2.cvtColor(frame_bw, cv2.COLOR_BGR2HSV)
            lo = np.array([106, 0, 250], dtype=np.uint8)
            hi = np.array([180, 9, 255], dtype=np.uint8)
            bw_mask = cv2.inRange(hsv, lo, hi)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bw_mask = cv2.dilate(bw_mask, kernel, iterations=2)
            bw_mask = cv2.erode(bw_mask, kernel, iterations=1)

            # 选中距离相机中心最近的亮蓝目标
            bw_cnts, _ = cv2.findContours(bw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_bw_dist = float("inf")
            for cnt in bw_cnts:
                # 多边形逼近 → 中心
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                M = cv2.moments(approx)
                if M["m00"] > 0:
                    bcx = int(M["m10"] / M["m00"])
                    bcy = int(M["m01"] / M["m00"])
                else:
                    M_raw = cv2.moments(cnt)
                    bcx = int(M_raw["m10"] / M_raw["m00"]) if M_raw["m00"] > 0 else 0
                    bcy = int(M_raw["m01"] / M_raw["m00"]) if M_raw["m00"] > 0 else 0

                dist = abs(bcx - cam_x) + abs(bcy - cam_y)
                if dist < best_bw_dist:
                    best_bw_dist = dist
                    best_bw_center = (bcx, bcy)

            if best_bw_center is not None:
                # 紫点存在 → 锁定 + 发送紫点 vs 框中心的偏差
                bw_locked = True
                cv2.circle(feat["vis"], best_bw_center, 8, (255, 0, 255), -1)
                ptz.send_raw(best_bw_center[0] - fc_x, best_bw_center[1] - fc_y)
            else:
                # 无亮蓝目标 → 解锁 + 回退
                bw_locked = False
                ptz.send_deviation(fc_x, fc_y)
        else:
            ptz.send_deviation(fc_x, fc_y)
    else:
        bw_locked = False
        ptz.send_no_target()

    # ---- 偏差连线与右上角文字 (紫点优先) ----
    if best_bw_center is not None:
        fc_x, fc_y = feat["frame_center"]
        wx = best_bw_center[0] - fc_x
        wy = best_bw_center[1] - fc_y
        cv2.line(feat["vis"], best_bw_center, (int(fc_x), int(fc_y)), (255, 0, 255), 2)
        vis_w = feat["vis"].shape[1]
        cv2.putText(feat["vis"], f"wx: {wx:+.1f} px (BW)", (vis_w - 260, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(feat["vis"], f"wy: {wy:+.1f} px (BW)", (vis_w - 260, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    elif feat["frame_center"] is not None:
        fc_x, fc_y = feat["frame_center"]
        cam_x, cam_y = actual_w / 2, actual_h / 2
        wx = cam_x - fc_x
        wy = cam_y - fc_y
        cv2.line(feat["vis"], (int(cam_x), int(cam_y)), (int(fc_x), int(fc_y)), (0, 255, 255), 2)
        vis_w = feat["vis"].shape[1]
        cv2.putText(feat["vis"], f"wx: {wx:+.1f} px", (vis_w - 260, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(feat["vis"], f"wy: {wy:+.1f} px", (vis_w - 260, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ---- FPS 统计 ----
    new_stats = ptz.tick()
    if new_stats is not None:
        stats = new_stats

    # 左上角显示 FPS / 发送频率
    if stats is not None:
        vis = feat["vis"]
        fps_texts = [
            f"FPS:     {stats[0]:5.1f}",
            f"Serial:  {stats[1]:5.1f} Hz",
            f"NoTarget:{stats[2]:5.1f} Hz",
        ]
        for i, txt in enumerate(fps_texts):
            cv2.putText(vis, txt, (10, 28 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     show_resized("Black Boxes", feat["vis"], 0.5)
#     show_resized("Binary", feat["binary"], 0.5)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
#     elif key == ord("="):
#         threshold_val = min(255, threshold_val + 5)
#         print("threshold_val =", threshold_val)
#     elif key == ord("-"):
#         threshold_val = max(0, threshold_val - 5)
#         print("threshold_val =", threshold_val)

ptz.close()
cap.release()
# cv2.destroyAllWindows()
