
A4纸 PnP 估计器模块
===============

基于 A4 纸（扣除胶带后）的矩形检测、单应性平面测距、solvePnP 深度估计。
支持亚像素角点精化、透视裁切、面积比计算和调试可视化。

类:
    A4CalibEstimator
        封装完整的 A4 纸检测与 PnP 估计管线。
        加载 camera_params.yaml 内参，处理 BGR 帧，返回检测结果字典。

用法:
    est = A4CalibEstimator("camera_params.yaml", use_undistort=False)
    result = est.process(frame)  # frame: BGR
    vis = est.draw_debug(result["undist"], result)
"""

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """对四点排序：左上/右上/右下/左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class A4CalibEstimator:
    """
    A4纸检测 + 平面尺度(单应性) + 深度(Z)(solvePnP) 的封装类。

    参数:
        yaml_path: 相机内参 YAML 文件路径 (camera_params.yaml)
        a4_size_mm: A4 纸尺寸 (宽, 高) mm, 默认 (210.0, 297.0)
        tape_width_mm: 胶带宽度 mm, 从 A4 边上扣除
        aspect_tol: 长宽比容差, 默认 ±8%
        use_undistort: 是否对输入帧去畸变
        undistort_newK: 去畸变时是否使用最优内参矩阵

    用法:
        est = A4CalibEstimator("camera_params.yaml", use_undistort=False)
        result = est.process(frame)  # frame: BGR
    """

    def __init__(
        self,
        yaml_path: str = "camera_params.yaml",
        a4_size_mm=(210.0, 297.0),
        tape_width_mm: float = 46.4,
        aspect_tol: float = 0.08,
        use_undistort: bool = False,
        undistort_newK: bool = False,
    ):
        # 读取标定参数
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Cannot open yaml file: {yaml_path}")

        self.K = fs.getNode("camera_matrix").mat()
        self.dist = fs.getNode("dist_coeff").mat()
        fs.release()

        if self.K is None or self.dist is None:
            raise ValueError("camera_matrix or dist_coeff not found in yaml.")

        # === A4 实际尺寸（扣除胶带）===
        W0, H0 = float(a4_size_mm[0]), float(a4_size_mm[1])
        self.W = W0 - float(tape_width_mm)
        self.H = H0 - float(tape_width_mm)

        self.ASPECT_RATIO = self.H / self.W
        self.ASPECT_TOL = float(aspect_tol)

        # 3D 点（solvePnP）中心
        self.obj_pts_3d = np.array(
            [
                [-self.W / 2, -self.H / 2, 0],  # 左上
                [self.W / 2, -self.H / 2, 0],  # 右上
                [self.W / 2, self.H / 2, 0],  # 右下
                [-self.W / 2, self.H / 2, 0],  # 左下
            ],
            dtype="float32",
        )

        # 2D 点（单应性）
        self.obj_pts_2d = np.array([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]], dtype="float32")

        self.use_undistort = bool(use_undistort)
        self.undistort_newK = bool(undistort_newK)

        # cornerSubPix 参数
        self.corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.corner_win = (5, 5)

        # undistort map（首次拿到frame尺寸后初始化）
        self._map1 = None
        self._map2 = None
        self._undist_size = None
        self._newK = None

    def _ensure_undistort_maps(self, frame_shape):
        h, w = frame_shape[:2]
        if self._undist_size == (w, h) and self._map1 is not None:
            return

        if self.undistort_newK:
            self._newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
        else:
            self._newK = self.K.copy()

        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self._newK, (w, h), cv2.CV_16SC2
        )
        self._undist_size = (w, h)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if not self.use_undistort:
            return frame
        self._ensure_undistort_maps(frame.shape)
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    def find_best_quad(
        self,
        gray: np.ndarray,
        image_center_x: int,
        image_center_y: int,
        min_area: float = 5000,
        max_area_ratio: float = 0.75,
        edge_margin_ratio: float = 0.03,
    ):
        """
        用多种阈值策略寻找最接近画面中心、符合 A4 长宽比的四边形。

        返回：
            (img_pts(4x2 float32, 已排序), th_display) 或 (None, None)
        """
        h_img, w_img = gray.shape
        frame_area = h_img * w_img
        max_area = frame_area * max_area_ratio
        edge_margin = min(h_img, w_img) * edge_margin_ratio

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        candidate_th = []

        _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidate_th.append(th_otsu)
        candidate_th.append(cv2.bitwise_not(th_otsu))

        th_adapt = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
        )
        candidate_th.append(th_adapt)

        edges = cv2.Canny(blur, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        th_canny = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidate_th.append(th_canny)

        best_quads = []  # (dist_to_center, ordered_pts)

        for th in candidate_th:
            for retr in (cv2.RETR_LIST, cv2.RETR_EXTERNAL):
                cnts, _ = cv2.findContours(th, retr, cv2.CHAIN_APPROX_SIMPLE)

                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < min_area or area > max_area:
                        continue

                    peri = cv2.arcLength(c, True)
                    approx = None
                    for eps_factor in (0.02, 0.03, 0.04, 0.015, 0.05):
                        ap = cv2.approxPolyDP(c, eps_factor * peri, True)
                        if len(ap) == 4:
                            approx = ap
                            break
                    if approx is None:
                        continue

                    pts = approx.reshape(4, 2).astype("float32")
                    ordered = _order_points(pts)

                    # 角点别贴边
                    if (
                        np.any(ordered[:, 0] < edge_margin)
                        or np.any(ordered[:, 0] > w_img - edge_margin)
                        or np.any(ordered[:, 1] < edge_margin)
                        or np.any(ordered[:, 1] > h_img - edge_margin)
                    ):
                        continue

                    # 长宽比过滤 + 边长一致性
                    w1 = np.linalg.norm(ordered[1] - ordered[0])
                    w2 = np.linalg.norm(ordered[2] - ordered[3])
                    h1 = np.linalg.norm(ordered[3] - ordered[0])
                    h2 = np.linalg.norm(ordered[2] - ordered[1])
                    if min(w1, w2, h1, h2) < 10:
                        continue
                    if min(w1, w2) / max(w1, w2) < 0.70 or min(h1, h2) / max(h1, h2) < 0.70:
                        continue

                    w_px = (w1 + w2) / 2
                    h_px = (h1 + h2) / 2
                    ratio = max(w_px, h_px) / min(w_px, h_px)
                    if not (self.ASPECT_RATIO * (1 - self.ASPECT_TOL) <= ratio <= self.ASPECT_RATIO * (1 + self.ASPECT_TOL)):
                        continue

                    # 凸性/贴合度
                    hull = cv2.convexHull(approx)
                    hull_area = cv2.contourArea(hull)
                    if hull_area < 1 or area / hull_area < 0.85:
                        continue

                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    d = ((cX - image_center_x) ** 2 + (cY - image_center_y) ** 2) ** 0.5

                    best_quads.append((d, ordered))

        if not best_quads:
            return None, None

        best_quads.sort(key=lambda x: x[0])
        return best_quads[0][1], candidate_th[0]  # th_display 仅用于调试显示

    def _refine_corners_subpix(self, gray: np.ndarray, img_pts: np.ndarray) -> np.ndarray:
        pts = img_pts.astype(np.float32).copy()
        return cv2.cornerSubPix(gray, pts, self.corner_win, (-1, -1), self.corner_criteria)

    def process(
        self,
        frame: np.ndarray,
        min_area: float = 5000,
        max_area_ratio: float = 0.75,
        edge_margin_ratio: float = 0.03,
        measure_half_span_px: int = 100,
        refine_subpix: bool = True,
    ) -> dict:
        """
        输入：frame(BGR)
        输出：dict，包含是否检测到、四点、单应性距离、Z等

        返回字段：
          detected: bool
          undist: 去畸变后的图(如果use_undistort=True)
          quad: (4,2) float32 排序四点 (如果 detected)
          H: 3x3 单应性矩阵 (如果可算)
          plane_dist_mm: 在画面中心左右 measure_half_span_px*2 像素对应的实际距离(mm) (如果可算)
          z_mm: solvePnP 求得 Z (mm) (如果可算)
          rvec, tvec: solvePnP 输出
          th_display: 二值调试图(可为空)
          center: (cx,cy)
        """
        undist = self.undistort(frame)
        h, w = undist.shape[:2]
        cx, cy = w // 2, h // 2

        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        img_pts, th_display = self.find_best_quad(
            gray, cx, cy, min_area=min_area, max_area_ratio=max_area_ratio, edge_margin_ratio=edge_margin_ratio
        )
        if img_pts is None:
            return {
                "detected": False,
                "undist": undist,
                "quad": None,
                "a4_frame": None,
                "H": None,
                "plane_dist_mm": None,
                "z_mm": None,
                "rvec": None,
                "tvec": None,
                "th_display": th_display,
                "center": (cx, cy),
            }

        if refine_subpix:
            img_pts = self._refine_corners_subpix(gray, img_pts)

        a4_frame = self.warp_a4_frame(undist, img_pts)

        # 单应性：平面距离
        Hmat, _ = cv2.findHomography(img_pts, self.obj_pts_2d)
        plane_dist_mm = None
        if Hmat is not None:
            p1_img = np.array([[cx - measure_half_span_px, cy]], dtype="float32")
            p2_img = np.array([[cx + measure_half_span_px, cy]], dtype="float32")
            p1_real = cv2.perspectiveTransform(p1_img[None], Hmat)[0][0]
            p2_real = cv2.perspectiveTransform(p2_img[None], Hmat)[0][0]
            plane_dist_mm = float(np.linalg.norm(p1_real - p2_real))

        # solvePnP：Z 深度
        z_mm, rvec, tvec = None, None, None
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_pts_3d, img_pts, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE
        )
        if ok:
            z_mm = float(tvec[2][0])

        return {
            "detected": True,
            "undist": undist,
            "quad": img_pts,
            "a4_frame": a4_frame,
            "H": Hmat,
            "plane_dist_mm": plane_dist_mm,
            "z_mm": z_mm,
            "rvec": rvec,
            "tvec": tvec,
            "th_display": th_display,
            "center": (cx, cy),
        }

    def compute_area_ratio(self, quad: np.ndarray) -> float:
        """
        根据检测到的 A4 四点，计算面积比例：现实面积(mm²) / 像素面积(px²)
        quad: (4,2) 排序后的四点 [左上, 右上, 右下, 左下]
        """
        ordered = quad
        w1 = np.linalg.norm(ordered[1] - ordered[0])
        w2 = np.linalg.norm(ordered[2] - ordered[3])
        h1 = np.linalg.norm(ordered[3] - ordered[0])
        h2 = np.linalg.norm(ordered[2] - ordered[1])
        w_px = (w1 + w2) / 2
        h_px = (h1 + h2) / 2
        pixel_area = w_px * h_px
        real_area = self.W * self.H
        return real_area / pixel_area

    @staticmethod
    def draw_debug(
        img_bgr: np.ndarray,
        result: dict,
        draw_center: bool = True,
        draw_quad: bool = True,
        show_text: bool = True,
    ) -> np.ndarray:
        """把检测结果画到图上（不修改原图，返回副本）"""
        out = img_bgr.copy()
        cx, cy = result.get("center", (out.shape[1] // 2, out.shape[0] // 2))

        if draw_center:
            cv2.circle(out, (cx, cy), 5, (255, 0, 0), -1)

        if result.get("detected") and draw_quad:
            quad = result["quad"]
            cv2.polylines(out, [quad.astype(np.int32)], True, (0, 255, 255), 2)
            for p in quad.astype(int):
                cv2.circle(out, tuple(p), 8, (0, 0, 255), -1)

        if show_text:
            if not result.get("detected"):
                cv2.putText(out, "A4 not detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if result.get("plane_dist_mm") is not None:
                    cv2.putText(out, f"Plane Dist: {result['plane_dist_mm']:.2f} mm",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if result.get("z_mm") is not None:
                    cv2.putText(out, f"Z Depth: {result['z_mm']:.2f} mm",
                                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        return out

    def warp_a4_frame(
            self,
            frame_bgr: np.ndarray,
            quad: np.ndarray,
            out_size_px: tuple[int, int] | None = None,
            keep_aspect: bool = True,
            border_value=(0, 0, 0),
    ) -> np.ndarray | None:
        """
        使用检测到的A4四点，把A4区域透视拉正并裁剪成一个矩形frame输出。

        frame_bgr: 原始BGR图（建议传 undist 后的那张）
        quad: (4,2) float32，顺序 [左上, 右上, 右下, 左下]
        out_size_px:
            - None: 自动按当前检测到的像素宽高输出
            - (w, h): 强制输出固定大小
        keep_aspect:
            - True 且 out_size_px 给了：会按A4比例调整输出尺寸（避免变形）
        border_value: warpPerspective 填充边界的颜色
        """
        q = quad.astype(np.float32)

        w1 = np.linalg.norm(q[1] - q[0])
        w2 = np.linalg.norm(q[2] - q[3])
        h1 = np.linalg.norm(q[3] - q[0])
        h2 = np.linalg.norm(q[2] - q[1])
        src_w = float((w1 + w2) / 2.0)
        src_h = float((h1 + h2) / 2.0)

        if src_w < 2 or src_h < 2:
            return None

        if out_size_px is None:
            out_w = int(round(src_w))
            out_h = int(round(src_h))
        else:
            out_w, out_h = int(out_size_px[0]), int(out_size_px[1])
            if keep_aspect:
                target_ratio = self.H / self.W
                cur_ratio = out_h / max(out_w, 1)
                if abs(cur_ratio - target_ratio) > 1e-6:
                    out_h = int(round(out_w * target_ratio))

        out_w = max(out_w, 2)
        out_h = max(out_h, 2)

        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(q, dst)
        warped = cv2.warpPerspective(
            frame_bgr,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return warped