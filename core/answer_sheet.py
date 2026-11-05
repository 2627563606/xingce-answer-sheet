# -*- coding: utf-8 -*-
"""
行测答题卡识别核心模块（OpenCV）

识别流程（严格遵循需求要点）：
  1. 预处理：灰度化 → 高斯滤波 → 自适应阈值(adaptiveThreshold，抗光照不均)
  2. 答题卡定位：轮廓面积筛选取最大四边形 → cv2.approxPolyDP 获取四角 → 透视矫正
  3. 气泡检测：轮廓 + 中值半径滤波 + 质心去重
  4. 题号排列：纵向“S”型（题号竖排、蛇形走位），自动检测每列题数/列数并正确映射题号
  5. 填涂判断：计算选项圈内涂黑像素占比，超过 fill_threshold(默认 35%) 判为选中
  6. 附加识别：准考证号（10-12 位，每位 0-9）与 AB 卷代码区

排列异常时会给出 warning 提示用户手动确认；界面端另提供手动修正模式。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import OPTIONS

# BGR 颜色
_GREEN = (0, 200, 0)
_RED = (0, 0, 255)
_GRAY = (160, 160, 160)
_BLUE = (255, 0, 0)


@dataclass
class Bubble:
    """一个填涂圈/填涂框（气泡）。"""
    x: float
    y: float
    r: float
    fill: float = 0.0          # 涂黑像素占比 0~1
    hole: bool = False         # True=空心(未填涂的圈/框)；False=实心候选
    bx: float = 0.0            # 外轮廓包围盒（用于实心块 fill 计算）
    by: float = 0.0
    bw: float = 0.0
    bh: float = 0.0


@dataclass
class SheetResult:
    """整张答题卡的识别结果。"""
    answers: Dict[int, str] = field(default_factory=dict)          # {题号: 选项}
    fill_map: Dict[int, Dict[str, float]] = field(default_factory=dict)   # {题号: {选项: 占比}}
    positions: Dict[int, Dict[str, Tuple[float, float]]] = field(default_factory=dict)  # {题号: {选项: (x,y)}}
    registration: str = ""       # 准考证号
    paper_code: str = ""         # AB 卷代码
    arrangement: str = "vertical_s"
    rows_per_band: int = 0       # 每列题数
    band_count: int = 0          # 列数
    detected_questions: int = 0  # 检测到的总题数
    warning: str = ""
    warped: Optional[np.ndarray] = None
    bubbles: List[Bubble] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


class AnswerSheetRecognizer:
    """行测答题卡识别器。

    参数（均可按实际版式微调）：
      fill_threshold:  选中判定阈值，默认 0.35（行测涂卡要求“深满匀”，阈值略高）
      min_rows_per_band: 每列最少行数，低于则视为异常排列
      column_order:   列序 'ltr'(左→右) / 'rtl'(右→左)
      start_direction: 最左列读数方向 1(自上而下) / -1(自下而上)，用于 S 型蛇形映射
      digit_order:    准考证号每位从上到下对应的数字序列，默认 0-9
      registration_len: 期望准考证号位数（0 表示不限制，用于校验提示）
      max_image_size: 输入长边上限，超限等比缩小以提速
    """

    def __init__(
        self,
        fill_threshold: float = 0.35,
        min_rows_per_band: int = 3,
        column_order: str = "ltr",
        start_direction: int = 1,
        digit_order: Optional[List[int]] = None,
        registration_len: int = 0,
        max_image_size: int = 2200,
        layout: str = "auto",
        block_seg: int = 5,
    ):
        self.fill_threshold = fill_threshold
        self.min_rows_per_band = min_rows_per_band
        self.column_order = column_order
        self.start_direction = start_direction
        self.digit_order = digit_order if digit_order is not None else list(range(10))
        self.registration_len = registration_len
        self.max_image_size = max_image_size
        # 版式：'s'=连续蛇形(每列从1开始连续编号)；'block'=分块式(每 block_seg 题一段、
        # 段间跳 nband*block_seg)；'auto'=按 列数/每列行数 启发式自动判定
        self.layout = layout
        self.block_seg = block_seg

    # ------------------------------------------------------------------ #
    # 对外主入口
    # ------------------------------------------------------------------ #
    def recognize(self, image: np.ndarray) -> SheetResult:
        res = SheetResult()

        # 1) 缩放 + 灰度 + 高斯滤波
        img = image
        h, w = img.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2) 定位答题卡四角并透视矫正
        warped_bgr, warped_gray, quad = self._detect_and_warp(img, gray)
        if warped_gray is None:
            res.warning = "未检测到答题卡四边形轮廓，请确认照片完整、答题卡在画面中且光线充足。"
            return res
        res.warped = warped_bgr
        res.debug["quad"] = None if quad is None else quad.tolist()

        # 3) 自适应阈值（抗光照不均）
        thresh = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10)
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # 4) 气泡检测
        bubbles = self._detect_bubbles(thresh)
        res.bubbles = bubbles
        if len(bubbles) < 30:
            res.warning = f"检测到的填涂点过少({len(bubbles)})，请检查图片质量、角度或答题卡版式。"
            return res

        # 5) 用 y 直方图分割「答案区 / 表头区」，再对答案区做列聚类 → 识别纵向 S 型题号矩阵
        ans_lo, ans_hi = self._answer_y_range(bubbles, warped_gray.shape[0])
        res.debug["answer_y_range"] = (round(ans_lo, 1), round(ans_hi, 1))
        ans_bubbles = [b for b in bubbles if ans_lo <= b.y <= ans_hi]
        hdr_bubbles = [b for b in bubbles if b.y < ans_lo or b.y > ans_hi]

        # 5a) 先尝试「分块式网格」版式（真实行测卡：4 区块 × 每区块=题号列+4 选项列，
        #      每 5 题一段纵向堆叠；选项为「方框+框内字母」，须用整格自适应阈值判填涂）
        block_out = self._try_block_grid(warped_gray, thresh, ans_bubbles)
        if block_out is not None:
            answers, fill_map, positions, rpb, nband = block_out
            res.answers = answers
            res.fill_map = fill_map
            res.positions = positions
            res.rows_per_band = rpb
            res.band_count = nband
            res.detected_questions = rpb * nband
            res.arrangement = "block"
            # 表头通常被透视裁剪，准考证号尽力识别（AB卷功能已按用户要求移除）
            candidate_meta = hdr_bubbles
            digit_cols = self._detect_digit_columns(candidate_meta)
            res.registration = self._read_digits(digit_cols)
            res.paper_code = ""
            res.warning = self._validate(res)
            return res

        groups = self._cluster_bubbles_by_x(ans_bubbles)
        chain = self._detect_bands(groups)
        if not chain:
            res.warning = (
                f"未能识别出规范的题号矩阵（每列≥{self.min_rows_per_band}行 × 4 选项），"
                "请确认答题卡为纵向S型排列；若版式特殊请使用手动修正模式。"
            )
            return res

        answers, fill_map, positions, rpb, nband, layout_used = self._extract_answers(chain)
        res.answers = answers
        res.fill_map = fill_map
        res.positions = positions
        res.rows_per_band = rpb
        res.band_count = nband
        res.detected_questions = rpb * nband
        res.arrangement = "block" if layout_used == "block" else "vertical_s"

        # 6) 准考证号（表头区 + 答案区未被题号带使用的气泡；AB卷功能已按用户要求移除）
        used = set()
        for start, _rows in chain:
            used.update(range(start, start + 4))
        side_nonband = [b for gi, g in enumerate(groups) if gi not in used for b in g]
        res.debug["nonband_count"] = len(side_nonband)
        candidate_meta = hdr_bubbles + side_nonband

        digit_cols = self._detect_digit_columns(candidate_meta)
        res.registration = self._read_digits(digit_cols)
        res.paper_code = ""

        # 7) 异常校验
        res.warning = self._validate(res)
        return res

    # ------------------------------------------------------------------ #
    # 预处理 / 透视矫正
    # ------------------------------------------------------------------ #
    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """四角排序：左上、右上、右下、左下。"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        (tl, tr, br, bl) = pts
        widthA = float(np.linalg.norm(br - bl))
        widthB = float(np.linalg.norm(tr - tl))
        max_width = max(int(round(widthA)), int(round(widthB)))
        heightA = float(np.linalg.norm(tr - br))
        heightB = float(np.linalg.norm(tl - bl))
        max_height = max(int(round(heightA)), int(round(heightB)))
        if max_width < 10 or max_height < 10:
            raise ValueError("透视变换目标尺寸过小")
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def _detect_and_warp(self, img, gray):
        """Canny 边缘 → 面积筛选最大轮廓 → approxPolyDP 四角 → 透视矫正。

        若答题卡四角被照片边缘裁切（无闭合四边形），退化为 Otsu 分割取最大亮区
        的最小外接矩形作为四角，保证仍能矫正。
        """
        h, w = gray.shape
        edged = cv2.Canny(gray, 50, 150)
        edged = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        quad = None
        min_area = h * w * 0.15
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                break
        if quad is None:
            quad = self._otsu_sheet_quad(gray)
        if quad is None:
            return None, None, None
        try:
            quad = self._order_points(quad)
            warped_gray = self._four_point_transform(gray, quad)
            warped_bgr = self._four_point_transform(img, quad)
        except Exception:
            return None, None, None
        return warped_bgr, warped_gray, quad

    @staticmethod
    def _otsu_sheet_quad(gray: np.ndarray):
        """Otsu 二值化 → 最大亮区 → 最小外接矩形四角（应对四角被裁的场景）。"""
        try:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < gray.shape[0] * gray.shape[1] * 0.1:
                return None
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            return box.astype("float32")
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # 气泡检测
    # ------------------------------------------------------------------ #
    def _detect_bubbles(self, thresh: np.ndarray) -> List[Bubble]:
        """检测填涂圈/框。

        用 RETR_CCOMP 只取「外轮廓」：
          - 空心圈/空心框（未填涂）→ 外轮廓带内孔 → hole=True；
          - 实心填涂块 → 外轮廓无内孔 → hole=False。
        这样可同时适配「圆形气泡卡」与「方框填涂卡」：空心直接判为未涂，
        实心块用包围盒内缩区域算涂黑占比（避开描边/边框）。
        """
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        hi = hierarchy[0]
        raw: List[dict] = []
        for i, cnt in enumerate(contours):
            if hi[i][3] != -1:          # 只取外轮廓（跳过内孔）
                continue
            area = cv2.contourArea(cnt)
            if area < 1:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 1 or bh < 1:
                continue
            aspect = bw / bh
            if aspect < 0.4 or aspect > 2.5:     # 排除长条文字/噪声
                continue
            m = cv2.moments(cnt)
            if m["m00"] <= 0:
                continue
            raw.append({
                "x": m["m10"] / m["m00"], "y": m["m01"] / m["m00"],
                "r": math.sqrt(area / math.pi), "area": area,
                "bx": x, "by": y, "bw": bw, "bh": bh,
                "hole": (hi[i][2] != -1),
            })

        if not raw:
            return []
        radii = np.array([t["r"] for t in raw])
        med = float(np.median(radii))
        cands = [t for t in raw if med * 0.35 <= t["r"] <= med * 2.2]

        # 实心块（无内孔）须与「空心框」尺寸相当，排除圈线断裂碎片/印刷文字
        hole_items = [t for t in cands if t["hole"]]
        if hole_items:
            fw = float(np.median([t["bw"] for t in hole_items]))
            fh = float(np.median([t["bh"] for t in hole_items]))
        else:
            fw = fh = float(np.median([max(t["bw"], t["bh"]) for t in cands]))
        keep = []
        for t in cands:
            if not t["hole"]:
                if not (0.55 * fw <= t["bw"] <= 1.7 * fw and 0.55 * fh <= t["bh"] <= 1.7 * fh):
                    continue
            keep.append(t)

        # 按 x 排序后质心合并：圈线断裂弧段并回所在圈/框（保留面积大的轮廓）
        keep.sort(key=lambda t: t["x"])
        merged: List[dict] = []
        merge_d = max(med * 1.1, 4.0)
        for t in keep:
            placed = False
            for g in merged:
                if math.hypot(t["x"] - g["x"], t["y"] - g["y"]) <= merge_d:
                    if t["area"] > g["area"]:
                        g.update(t)
                    g["hole"] = g["hole"] or t["hole"]
                    placed = True
                    break
            if not placed:
                merged.append(dict(t))

        # 面积离群过滤：相邻带气泡若被桥接成“合并块”，面积约为正常的 3 倍，丢弃
        if merged:
            areas = np.array([t["area"] for t in merged])
            med_area = float(np.median(areas))
            merged = [t for t in merged if t["area"] <= max(med_area * 2.5, 50.0)]

        bubbles: List[Bubble] = []
        for t in merged:
            b = Bubble(x=t["x"], y=t["y"], r=t["r"], hole=t["hole"],
                       bx=t["bx"], by=t["by"], bw=t["bw"], bh=t["bh"])
            # 一律用包围盒内缩区域算涂黑占比：空心框/圈内缩后≈0，
            # 实心填涂（含“环+中心块”的填涂圆）内缩后≈1。
            b.fill = self._fill_bbox(thresh, b)
            bubbles.append(b)
        return bubbles

    @staticmethod
    def _fill_bbox(thresh: np.ndarray, b: Bubble) -> float:
        """实心块：包围盒四边内缩 25% 后的区域涂黑占比。

        空心框的描边位于包围盒边缘，内缩后只剩框内空白（未涂≈0）或
        实心填涂（≈1），从而正确反映“深满匀”填涂。"""
        x, y, w, h = int(round(b.bx)), int(round(b.by)), int(round(b.bw)), int(round(b.bh))
        mgn = max(1, int(round(0.25 * min(w, h))))
        x0, y0 = max(0, x + mgn), max(0, y + mgn)
        x1, y1 = min(thresh.shape[1], x + w - mgn), min(thresh.shape[0], y + h - mgn)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return 0.0
        inner = thresh[y0:y1, x0:x1]
        total = float(inner.size)
        if total <= 0:
            return 0.0
        return float(cv2.countNonZero(inner)) / total

    @staticmethod
    def _cluster_ys(bubbles: List[Bubble], height: int) -> Tuple[List[Tuple[float, int]], float]:
        """把气泡按 y 聚成“行带”，返回 [(行带中心y, 气泡数)] 及行距容差。
        行距容差由「大于 0.3 倍半径的真实行间距」中值推导，避免列内抖动干扰。"""
        if not bubbles:
            return [], 1.0
        ys = sorted(b.y for b in bubbles)
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1e-6]
        med_r = float(np.median([b.r for b in bubbles]))
        real_gaps = [g for g in gaps if g > 0.3 * med_r]
        if real_gaps:
            y_tol = max(float(np.median(real_gaps)) * 0.55, 1.0)
        else:
            y_tol = med_r
        bands: List[List[float]] = [[ys[0]]]
        for v in ys[1:]:
            if v - bands[-1][-1] <= y_tol:
                bands[-1].append(v)
            else:
                bands.append([v])
        return [(float(np.mean(b)), len(b)) for b in bands], y_tol

    def _answer_y_range(self, bubbles: List[Bubble], height: int) -> Tuple[float, float]:
        """定位答案区纵向范围：答案行带的气泡数（=4选项×列数）显著多于表头行带
        （=数字列数）。用“排序后最大断档”把行带分成表头/答案两类，取答案类连续段。"""
        bands, y_tol = self._cluster_ys(bubbles, height)
        if len(bands) < self.min_rows_per_band:
            return (0.0, float(height))
        counts = sorted(n for _, n in bands)
        if len(counts) < 2:
            return (0.0, float(height))
        gaps = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]
        split = int(np.argmax(gaps))
        thr = (counts[split] + counts[split + 1]) / 2.0

        flagged = [n > thr for _, n in bands]
        idxs = [i for i, v in enumerate(flagged) if v]
        if not idxs:
            return (0.0, float(height))
        # 取「首尾被标记行带」覆盖的纵向区间：允许中间存在少量噪声行带，
        # 避免答案区被单行噪声截断（真实卡常见“分块式”版式段间会有杂散标记）
        y_lo = bands[idxs[0]][0] - y_tol
        y_hi = bands[idxs[-1]][0] + y_tol
        return (float(y_lo), float(y_hi))

    # ------------------------------------------------------------------ #
    # 列聚类 / S 型矩阵识别
    # ------------------------------------------------------------------ #
    @classmethod
    def _cluster_bubbles_by_x(cls, bubbles: List[Bubble], tol: float = None) -> List[List[Bubble]]:
        """按 x 坐标聚类成“列”。

        列间阈值取「有效列间距中值的 0.5 倍」与「0.8 倍气泡半径」的较大值：
          - 0.3 倍半径以下的微小间距视为同一列内抖动（阈值光晕噪声），先滤除；
          - 气泡中心距必须大于 2 倍半径才不会重叠，故 0.8r 阈值必然小于列间距。
        tol 参数可显式指定（分块式网格路径用固定容差，避免断裂轮廓导致列内分裂）。
        """
        if not bubbles:
            return []
        sb = sorted(bubbles, key=lambda b: b.x)
        gaps = [sb[i + 1].x - sb[i].x for i in range(len(sb) - 1) if sb[i + 1].x - sb[i].x > 1e-6]
        med_r = float(np.median([b.r for b in bubbles]))
        if not gaps:
            return [sb]
        if tol is None:
            sig_gaps = [g for g in gaps if g > 0.3 * med_r]
            if sig_gaps:
                tol = max(float(np.median(sig_gaps)) * 0.5, 0.8 * med_r)
            else:
                tol = med_r
        groups: List[List[Bubble]] = [[sb[0]]]
        for b in sb[1:]:
            if b.x - groups[-1][-1].x <= tol:
                groups[-1].append(b)
            else:
                groups.append([b])
        return groups

    @staticmethod
    def _estimate_y_tol(groups: List[List[Bubble]]) -> float:
        """由列内相邻 y 间距中值推导行聚类容差。"""
        diffs = []
        for g in groups:
            ys = sorted(b.y for b in g)
            for a, b in zip(ys, ys[1:]):
                if b - a > 1e-6:
                    diffs.append(b - a)
        if not diffs:
            return 5.0
        return max(float(np.median(diffs)) * 0.55, 1.0)

    def _align_rows(self, four: List[List[Bubble]], y_tol: float):
        """把 4 个选项列按 y 对齐成行。返回 [(行中值y, [4个气泡引用], 命中数)] 或 None。"""
        all_y: List[float] = []
        for g in four:
            all_y.extend(b.y for b in g)
        if not all_y:
            return None
        all_y.sort()
        levels: List[float] = []
        cur = [all_y[0]]
        for v in all_y[1:]:
            if v - cur[-1] <= y_tol:
                cur.append(v)
            else:
                levels.append(float(np.mean(cur)))
                cur = [v]
        levels.append(float(np.mean(cur)))

        rows = []
        for lev in levels:
            refs: List[Optional[Bubble]] = []
            for g in four:
                best, best_d = None, y_tol
                for b in g:
                    d = abs(b.y - lev)
                    if d <= best_d:
                        best_d, best = d, b
                refs.append(best)
            present = sum(1 for r in refs if r is not None)
            if present >= 3:              # 该行至少命中 3 个选项
                rows.append((lev, refs, present))
        rows.sort(key=lambda t: t[0])
        if len(rows) < self.min_rows_per_band:
            return None
        return rows

    @staticmethod
    def _split_blocks(gxs: List[float]) -> List[List[int]]:
        """按列中心 x 的大间隙（>1.5×中位间隙）把列切分为区块。"""
        if len(gxs) < 2:
            return [list(range(len(gxs)))]
        gaps = [gxs[i + 1] - gxs[i] for i in range(len(gxs) - 1)]
        med = float(np.median(gaps))
        blocks: List[List[int]] = []
        cur = [0]
        for i, gp in enumerate(gaps):
            if gp > med * 1.5:
                blocks.append(cur)
                cur = [i + 1]
            else:
                cur.append(i + 1)
        blocks.append(cur)
        return blocks

    def _try_block_grid(self, gray, thresh, ans_bubbles):
        """尝试识别「分块式网格」版式（真实行测答题卡）。

        特征：横向分为多个区块（本实现面向 4 区块 × 每区块 35 行），每区块首列为题号列、
        其后 4 列为 A-D 选项列；每 5 题一段纵向堆叠，段间留白。选项为「方框+框内印刷字母」，
        填涂时把框涂满（铅笔/马克笔，可能偏小偏上），因此用「整格自适应阈值黑占比」判填涂，
        不用圆形掩膜（那仅对样例的圆形气泡有效）。

        返回 (answers, fill_map, positions, rows_per_band, nband) 或 None（非分块式版式）。
        """
        # 行聚类
        bands, _y_tol = self._cluster_ys(ans_bubbles, gray.shape[0])
        row_ys = [y for y, _c in bands]
        if len(row_ys) < self.min_rows_per_band:
            return None
        # 列聚类（固定容差：方框+字母被拆碎时质心抖动可达 ~10px）
        groups = self._cluster_bubbles_by_x(ans_bubbles, tol=12.0)
        gxs = [float(np.median([b.x for b in g])) for g in groups]
        if len(gxs) < 8:
            return None
        blocks = self._split_blocks(gxs)
        col_counts = [len(b) for b in blocks]
        # 分块式判定：存在「题号列+4 选项列」的 5 列区块，且区块数>=2
        if len(blocks) < 2 or not any(c >= 5 for c in col_counts):
            return None
        # 每区块取后 4 列为选项列；非 5 列区块（异常断裂）则放弃
        opt_cols = []
        for blk in blocks:
            if len(blk) >= 5:
                opt_cols.extend(blk[-4:])
            else:
                return None
        opt_cols.sort()
        if len(opt_cols) % 4 != 0 or len(opt_cols) == 0:
            return None
        nblock = len(opt_cols) // 4
        seg = self.block_seg
        answers, fill_map, positions = {}, {}, {}
        halfy = max(10, int(round((row_ys[1] - row_ys[0]) * 0.45)) if len(row_ys) > 1 else 13)
        halfx = max(12, int(round((gxs[1] - gxs[0]) * 0.38)) if len(gxs) > 1 else 15)
        halfx = min(halfx, 18)
        for qi, col_idx in enumerate(opt_cols):
            xc = gxs[col_idx]
            blk = qi // 4
            opt_label = "ABCD"[qi % 4]
            for ri, yc in enumerate(row_ys):
                s, rr = divmod(ri, seg)
                q = (blk * seg + rr + 1) + (nblock * seg) * s
                y0, y1 = max(0, int(yc - halfy)), min(gray.shape[0], int(yc + halfy))
                x0, x1 = max(0, int(xc - halfx)), min(gray.shape[1], int(xc + halfx))
                reg = thresh[y0:y1, x0:x1]
                f = float(cv2.countNonZero(reg)) / reg.size if reg.size else 0.0
                fill_map.setdefault(q, {})[opt_label] = f
                positions.setdefault(q, {})[opt_label] = (round(xc, 1), round(yc, 1))
        for q, fm in fill_map.items():
            best = max("ABCD", key=lambda o: fm[o])
            if fm[best] >= self.fill_threshold:
                answers[q] = best
        return answers, fill_map, positions, len(row_ys), nblock

    def _detect_bands(self, groups: List[List[Bubble]]):
        """在全部列中寻找“每 4 列一组、行数一致”的最长 S 型题号带。"""
        n = len(groups)
        y_tol = self._estimate_y_tol(groups)
        cand: Dict[int, list] = {}
        for i in range(n - 3):
            rows = self._align_rows(groups[i:i + 4], y_tol)
            if rows:
                cand[i] = rows

        best: List[Tuple[int, list]] = []
        for s in cand:
            chain: List[Tuple[int, list]] = []
            i = s
            rpb = len(cand[s])
            while i in cand and len(cand[i]) == rpb:
                chain.append((i, cand[i]))
                i += 4
            if len(chain) > len(best):
                best = chain
        return best

    def _extract_answers(self, chain: List[Tuple[int, list]]):
        """按纵向 S 型 / 分块式 映射题号并判定选中项。

        返回 (answers, fill_map, positions, rpb, nband, layout_used)。
        layout_used: 's'=连续蛇形 / 'block'=分块式。
        """
        answers: Dict[int, str] = {}
        fill_map: Dict[int, Dict[str, float]] = {}
        positions: Dict[int, Dict[str, Tuple[float, float]]] = {}
        rpb = len(chain[0][1])
        nband = len(chain)

        # 版式判定：'auto' 时按「列数 vs 每列行数」启发式
        layout = self.layout
        if layout == "auto":
            if rpb > nband * 2 and rpb % self.block_seg == 0:
                layout = "block"
            else:
                layout = "s"

        col_iter = range(1, nband + 1)
        if self.column_order == "rtl":     # 列序从右往左
            col_iter = reversed(list(col_iter))

        for col_idx, (start, rows) in zip(col_iter, chain):
            if layout == "block":
                # 分块式：每 block_seg 题一段、段间跳 nband*block_seg，列内自上而下（同向）
                seg = self.block_seg
                for row_idx, (lev, refs, present) in enumerate(rows):
                    s = row_idx // seg
                    rr = row_idx % seg
                    qno = (col_idx - 1) * seg + rr + 1 + (nband * seg) * s
                    self._assign_row(fill_map, positions, answers, qno, refs)
                continue
            direction = self.start_direction
            if self.column_order == "rtl":
                direction = -direction
            for row_idx, (lev, refs, present) in enumerate(rows):
                if ((col_idx - 1) % 2 == 0 and direction == 1) or \
                   ((col_idx - 1) % 2 == 1 and direction == -1):
                    qno = (col_idx - 1) * rpb + row_idx + 1
                else:
                    qno = (col_idx - 1) * rpb + (rpb - row_idx)
                self._assign_row(fill_map, positions, answers, qno, refs)
        return answers, fill_map, positions, rpb, nband, layout

    def _assign_row(self, fill_map, positions, answers, qno, refs):
        fm: Dict[str, float] = {}
        pos: Dict[str, Tuple[float, float]] = {}
        for opt_i, ref in enumerate(refs):
            if ref is not None:
                fm[OPTIONS[opt_i]] = round(ref.fill, 4)
                pos[OPTIONS[opt_i]] = (round(ref.x, 1), round(ref.y, 1))
            else:
                fm[OPTIONS[opt_i]] = 0.0
        fill_map[qno] = fm
        positions[qno] = pos
        best_opt = max(OPTIONS, key=lambda o: fm[o])
        if fm[best_opt] >= self.fill_threshold:
            answers[qno] = best_opt

    # ------------------------------------------------------------------ #
    # 准考证号 / AB 卷
    # ------------------------------------------------------------------ #
    def _detect_digit_columns(self, nonband: List[Bubble]) -> List[List[Bubble]]:
        """从未进入题号矩阵的气泡中识别准考证号数字列（8~12 个、等距竖排）。"""
        if not nonband:
            return []
        groups = self._cluster_bubbles_by_x(nonband)
        cols = []
        for g in groups:
            ys = sorted(b.y for b in g)
            if not (8 <= len(ys) <= 12):
                continue
            gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            if not gaps:
                continue
            med = float(np.median(gaps))
            if med <= 1e-6:
                continue
            if max(abs(gp - med) for gp in gaps) > med * 0.6:   # 间距须均匀
                continue
            cols.append(sorted(g, key=lambda b: b.y))
        cols.sort(key=lambda c: float(np.mean([b.x for b in c])))
        return cols

    def _read_digits(self, digit_cols: List[List[Bubble]]) -> str:
        if not digit_cols:
            return ""
        digits = []
        for col in digit_cols:
            filled = [i for i, b in enumerate(col) if b.fill >= self.fill_threshold]
            if not filled:
                digits.append("")
            else:
                idx = max(filled, key=lambda i: col[i].fill)
                digits.append(str(self.digit_order[idx] % 10))
        return "".join(digits)

    def _detect_paper_code(self, bubbles: List[Bubble], exclude_ids: Optional[set] = None) -> str:
        """AB 卷：表头中相邻的一对气泡（一填一空），排除已归入准考证号的数字列气泡。"""
        exclude_ids = exclude_ids or set()
        bs = [b for b in bubbles if id(b) not in exclude_ids]
        if len(bs) < 2:
            return ""
        bs.sort(key=lambda b: b.x)
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                if abs(a.y - b.y) > max(a.r, b.r) * 3:
                    continue
                dx = abs(a.x - b.x)
                if not (0.5 * (a.r + b.r) <= dx <= 10 * (a.r + b.r)):
                    continue
                fa = a.fill >= self.fill_threshold
                fb = b.fill >= self.fill_threshold
                if fa and not fb:
                    return "A"
                if fb and not fa:
                    return "B"
                if fa and fb:
                    return "AB(双涂)"
        return ""

    # ------------------------------------------------------------------ #
    # 校验 / 可视化
    # ------------------------------------------------------------------ #
    def _validate(self, res: SheetResult) -> str:
        warns = []
        if res.detected_questions == 0:
            warns.append("未检测到任何题号。")
        if self.registration_len and res.registration:
            if len(res.registration) < self.registration_len:
                warns.append(f"准考证号识别为 {len(res.registration)} 位（期望 {self.registration_len} 位），"
                             "可能有位未涂或识别遗漏。")
        if self.registration_len and not res.registration:
            warns.append("未识别到准考证号。")
        return "；".join(warns)

    def draw_answers(self, result: SheetResult, standard: Optional[Dict[str, str]] = None) -> np.ndarray:
        """在矫正后图上绘制识别结果：选中圈绿=正确 / 红=错误 / 蓝=未涂判空。"""
        if result.warped is None:
            return np.full((300, 300, 3), 255, np.uint8)
        img = result.warped.copy()
        for qno, fm in result.fill_map.items():
            pos = result.positions.get(qno, {})
            best_opt = max(fm, key=fm.get)
            selected = fm[best_opt] >= self.fill_threshold
            for opt, fill in fm.items():
                if opt not in pos:
                    continue
                x, y = pos[opt]
                r = 12
                if selected and opt == best_opt:
                    correct = standard is not None and standard.get(str(qno), standard.get(qno)) == opt
                    color = _GREEN if correct else _RED
                    cv2.circle(img, (int(x), int(y)), r, color, 3)
                elif fill > 0.05 and fill < self.fill_threshold:
                    cv2.circle(img, (int(x), int(y)), r, _BLUE, 2)
                else:
                    cv2.circle(img, (int(x), int(y)), r, _GRAY, 1)
        return img
