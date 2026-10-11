import cv2
import numpy as np
import math
from itertools import combinations
from a4_cal import A4CalibEstimator

def show_resized(window_name, img, scale=0.5):
    show = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(window_name, show)

def extract_black_shape_vertices(
    a4_bgr,
    threshold_val=80,
    min_area=500,
    epsilon_ratio=0.02,
    do_morph=True,
):
    """
    在 A4 内部截图(a4_frame)里检测黑色几何形，并返回可视化结果。
    阶段1：灰度 + 二值(THRESH_BINARY_INV)
    阶段2：外轮廓 + approxPolyDP，红点画角点
    """
    vis = a4_bgr.copy()

    # ===== 阶段1：预处理 =====
    gray = cv2.cvtColor(a4_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    if do_morph:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)   # 去小噪点
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # 填小洞/断裂

    # ===== 阶段2：特征提取（轮廓 + approxPolyDP）=====
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertices_list = []  # 每个目标的顶点数组 (N,2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)  # (N,1,2)

        pts = approx.reshape(-1, 2)
        vertices_list.append(pts)

        # 可选：画轮廓（绿线）
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

        # 画角点（红点）
        for (x, y) in pts:
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)

        # 可选：标注顶点数
        x0, y0 = int(pts[0][0]), int(pts[0][1])
        cv2.putText(vis, f"verts={len(pts)} area={int(area)}",
                    (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return {
        "vis": vis,                 # 画了红点/轮廓的图
        "gray": gray,               # 灰度调试
        "binary": binary,           # 二值调试（黑物体变白）
        "vertices": vertices_list,  # 顶点列表
    }


def check_edge_type(binary, c1, c2, sample_count=20, side_offset=8, white_ratio_thresh=0.7):
    """
    检查边 c1->c2 是 EXPOSED（一侧前景/一侧背景）还是 INTERNAL。
    返回 (edge_type, inward_normal):
      edge_type    : "EXPOSED" | "INTERNAL" | "UNKNOWN"
      inward_normal: 指向物体内部的单位向量（仅 EXPOSED 时有效）
    """
    h, w = binary.shape[:2]
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    edge_vec = c2 - c1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-6:
        return "UNKNOWN", None

    edge_unit  = edge_vec / edge_len
    perp_left  = np.array([-edge_unit[1],  edge_unit[0]])   # 逆时针 90°
    perp_right = np.array([ edge_unit[1], -edge_unit[0]])   # 顺时针 90°

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
    bg_thresh   = 1.0 - white_ratio_thresh          # 另一侧必须够"黑"

    if left_ratio >= white_ratio_thresh and right_ratio < bg_thresh:
        return "EXPOSED", perp_left
    if right_ratio >= white_ratio_thresh and left_ratio < bg_thresh:
        return "EXPOSED", perp_right
    return "INTERNAL", None


def filter_square_vertices(binary, all_corners, angle_thresh=25):
    """
    筛选可能是正方形顶点的角点。
    对于每个角点，找到它与其他角点连接的所有 EXPOSED 边，
    如果存在两条边的夹角接近90度，则该角点可能是正方形顶点。

    参数:
        binary: 二值图像
        all_corners: 所有角点列表，每个元素为 (x, y) 或 np.array
        angle_thresh: 角度阈值（度），默认10度，即 80-100度算直角

    返回:
        符合条件的角点列表（np.array 格式）
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


def build_square_from_edge(c1, c2, inward_normal):
    """
    由暴露边 c1->c2 和朝内法向量构建正方形四顶点。
    顶点顺序（逆时针）：c1, c2, c2+inward*side, c1+inward*side
    """
    c1     = np.array(c1, dtype=float)
    c2     = np.array(c2, dtype=float)
    side   = np.linalg.norm(c2 - c1)
    inward = inward_normal / np.linalg.norm(inward_normal) * side
    c3 = c2 + inward
    c4 = c1 + inward
    return np.array([c1, c2, c3, c4], dtype=np.float32)


def validate_square_by_area_sampling(binary, square_pts, n_samples=50, white_thresh=0.9):
    """
    在候选正方形内部随机采样，判断是否是实心前景（白色比例 > white_thresh）。
    腐蚀约 5% 边长，避免边缘噪声干扰。
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


def are_squares_similar(sq1, sq2, center_dist_thresh=15, area_ratio_thresh=0.15):
    """中心距离近 且 面积相近 → 视为重复。"""
    def stats(sq):
        cx, cy = sq[:, 0].mean(), sq[:, 1].mean()
        area   = cv2.contourArea(sq.astype(np.float32))
        return np.array([cx, cy]), area

    c1, a1     = stats(sq1)
    c2, a2     = stats(sq2)
    dist       = np.linalg.norm(c1 - c2)
    area_ratio = abs(a1 - a2) / max(a1, a2, 1e-6)
    return dist < center_dist_thresh and area_ratio < area_ratio_thresh


def find_smallest_square(binary, valid_corners):
    """
    从 valid_corners 两两配对，寻找面积最小的真实正方形。
    返回 np.float32 的 (4,2) 顶点数组，找不到则返回 None。
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


est = A4CalibEstimator("camera_params.yaml", use_undistort=False)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

# 你可以先用键盘动态调阈值，快速找到合适的 threshold_val
threshold_val = 80

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = est.process(frame)

    debug = est.draw_debug(result["undist"], result)

    area_ratio = None
    if result["detected"] and result["quad"] is not None:
        area_ratio = est.compute_area_ratio(result["quad"])

    min_sq_area = None
    real_side = None

    if result.get("a4_frame") is not None:
        a4 = result["a4_frame"]
        show_resized("A4 Frame", a4, 0.5)

        feat = extract_black_shape_vertices(
            a4,
            threshold_val=threshold_val,
            min_area=600,
            epsilon_ratio=0.02,
            do_morph=True,
        )

        all_corners = [pt.astype(float) for pts in feat["vertices"] for pt in pts]
        valid_corners = filter_square_vertices(feat["binary"], all_corners, angle_thresh=10)

        vis_filtered = feat["vis"].copy()
        valid_set = [tuple(c.astype(int)) for c in valid_corners]
        for pts in feat["vertices"]:
            for (x, y) in pts:
                pt_int = (int(x), int(y))
                if pt_int in valid_set:
                    cv2.circle(vis_filtered, pt_int, 6, (255, 0, 0), -1)
                else:
                    cv2.circle(vis_filtered, pt_int, 6, (0, 0, 255), -1)

        show_resized("A4 Black Shapes (vis)", vis_filtered, 0.8)
        show_resized("A4 Binary (inv)", feat["binary"], 0.8)

        smallest = find_smallest_square(feat["binary"], valid_corners)

        if smallest is not None:
            vis_sq = a4.copy()
            pts_draw = smallest.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_sq, [pts_draw], isClosed=True, color=(0, 255, 255), thickness=3)
            min_sq_area = int(cv2.contourArea(smallest))
            if area_ratio is not None:
                real_area = min_sq_area * area_ratio
                real_side = math.sqrt(real_area)
            show_resized("Smallest Square", vis_sq, 0.8)
        else:
            blank = np.zeros_like(a4)
            show_resized("Smallest Square", blank, 0.8)

    if min_sq_area is not None:
        h, w = debug.shape[:2]
        if real_side is not None:
            cv2.putText(debug, f"min sq  side={real_side:.1f}mm",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(debug, f"min sq  area={min_sq_area}px",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    show_resized("Main", debug, 0.5)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("="):
        threshold_val = min(255, threshold_val + 5)
        print("threshold_val =", threshold_val)
    elif key == ord("-"):
        threshold_val = max(0, threshold_val - 5)
        print("threshold_val =", threshold_val)

cap.release()
cv2.destroyAllWindows()
