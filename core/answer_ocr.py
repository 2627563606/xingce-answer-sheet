# -*- coding: utf-8 -*-
"""
答案 OCR 模块：把拍照 / 扫描的答案解析成 {题号: "A/B/C/D"}。

两种来源：
  1. 图片（jpg/png）：答案照片 → OCR 识别文本 → 文本模式解析
  2. PDF 扫描版：pdfplumber 提不出文字时，逐页渲染成图 → OCR → 文本模式解析

双通道识别（同一份图片/扫描页做交叉验证）：
  A. 文本模式：OCR 文本按版面拼接 → 区间/逐条/对齐式文本解析
  B. 空间对齐：OCR 文本块带坐标，数字块 ↔ 字母块按 x 列对齐配对
     （针对两行式布局“题号行 + 答案行”，行序错乱也能配对）

依赖：rapidocr_onnxruntime（轻量 OCR 引擎，CPU 中英）、PyMuPDF（PDF 转图）。
引擎按需惰性加载（首次调用时初始化并下载模型，之后复用）。
"""
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

from .answer_parser import parse_answers_text

_ENGINE = None


def _get_engine():
    """惰性加载 OCR 引擎（单例）。"""
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _norm_wide(s: str) -> str:
    """全角字符转半角（OCR 常输出全角数字/字母）。"""
    out = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _ocr_result(data: bytes) -> Optional[List]:
    """引擎原始结果：[(box, text, score), ...]；失败/空 → None。"""
    engine = _get_engine()
    try:
        result, _ = engine(data)
    except Exception:
        return None
    return result or None


def _blocks_from_result(result) -> List[tuple]:
    """从原始结果提取 (x_center, y_center, x_min, x_max, text)。"""
    blocks = []
    for box, text, _score in result:
        try:
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
        except Exception:
            continue
        t = _norm_wide(str(text).strip())
        if t:
            blocks.append((sum(xs) / len(xs), sum(ys) / len(ys),
                           min(xs), max(xs), t))
    return blocks


def _lines_from_result(result) -> str:
    """把文本块按行聚类（行间容差 20px）、行内按 x 排序拼接。"""
    blocks = _blocks_from_result(result)
    if not blocks:
        return ""
    blocks.sort(key=lambda t: (t[1], t[0]))
    lines: List[List[tuple]] = []
    for x, y, _xm, _xx, text in blocks:
        if lines and abs(y - lines[-1][-1][0]) <= 20:
            lines[-1].append((x, text))
        else:
            lines.append([(x, text)])
    parts = []
    for line in lines:
        line.sort(key=lambda t: t[0])
        parts.append(" ".join(text for _x, text in line))
    return "\n".join(parts)


def ocr_image_bytes(data: bytes, dpi: int | None = None) -> str:
    """对图片字节 OCR，返回按版面顺序拼接的文本（兼容旧接口）。"""
    result = _ocr_result(data)
    return _lines_from_result(result) if result else ""


def _split_num_string(s: str) -> Optional[List[int]]:
    """把无分隔数字串按“递增题号序列”智能切分。

    例：'12345678910' → [1,2,3,4,5,6,7,8,9,10]；无法合理切分返回 None。
    """
    n = len(s)
    if n == 0:
        return None

    def dfs(i: int, prev: Optional[int], path: List[int]):
        if i == n:
            return path
        for L in (1, 2, 3):  # 优先 1 位（题号多为 1-2 位），再 2/3 位
            if i + L > n:
                continue
            v = int(s[i:i + L])
            if v == 0 or v > 999:
                continue
            if prev is not None and v <= prev:
                continue
            r = dfs(i + L, v, path + [v])
            if r is not None:
                return r
        return None

    return dfs(0, None, [])


def _align_blocks(blocks: List[tuple]) -> Dict[int, str]:
    """空间对齐配对：数字块/字母块（含连块）按 x 列对齐。

    规则：
      - 数字块（纯数字）→ 题号序列（无分隔连块自动切分为递增序列）
      - 字母块（纯 A-D）→ 字母序列
      - 字母块找 x 范围与之**重叠**的数字块，按序列顺序逐位配对：
        数字 [1,2,3,4,5] + 字母 [A,A,C,B,B] → 1:A 2:A 3:C 4:B 5:B
    用于两行式布局“题号行 + 答案行”在 OCR 行序错乱 / 连块时仍能正确映射。
    """
    num_seqs: List[tuple] = []   # (xmin, xmax, [q,...])
    let_seqs: List[tuple] = []   # (xmin, xmax, [c,...])
    for _x, _y, xmin, xmax, text in blocks:
        t = text.strip()
        nums = re.findall(r"\d{1,3}", t)
        lets = re.findall(r"[ABCDabcd]", t)
        if nums and not re.search(r"[A-Za-z]", t):
            qs = [int(n) for n in nums if 1 <= int(n) <= 999]
            if " " not in t and not re.search(r"[,，.．:：;；/\\\-]", t):
                spl = _split_num_string(t)
                if spl:
                    qs = spl
            if qs:
                num_seqs.append((xmin, xmax, qs))
        elif lets and not re.search(r"\d", t):
            cs = [c.upper() for c in lets]
            if cs:
                let_seqs.append((xmin, xmax, cs))
    out: Dict[int, str] = {}
    for lxmin, lxmax, cs in let_seqs:
        best, bestd = -1, 1e9
        for i, (dxmin, dxmax, qs) in enumerate(num_seqs):
            overlap = not (dxmax < lxmin or lxmax < dxmin)
            if not overlap:
                continue
            cd = abs((dxmin + dxmax) / 2 - (lxmin + lxmax) / 2)
            if cd < bestd:
                best, bestd = i, cd
        if best >= 0:
            qs = num_seqs[best][2]
            for idx, c in enumerate(cs):
                if idx < len(qs):
                    out.setdefault(qs[idx], c)
    return out


def _parse_ocr_result(result) -> Dict[int, str]:
    """一次 OCR 结果 → 合并文本模式 + 空间对齐两种解析。"""
    text = _lines_from_result(result)
    ans = parse_answers_text(text) if text else {}
    spatial = _align_blocks(_blocks_from_result(result))
    for q, c in spatial.items():
        if q not in ans:
            ans[q] = c
    return ans


def parse_answers_image(data: bytes) -> Dict[int, str]:
    """图片（拍照/扫描的答案）→ 答案字典（文本模式 + 空间对齐双通道）。"""
    result = _ocr_result(data)
    if not result:
        raise ValueError("OCR 未能从图片中识别出文字，请确认答案清晰、字体较大、无倾斜遮挡。")
    ans = _parse_ocr_result(result)
    if not ans:
        raise ValueError("图片已识别出文字，但未能提取出“题号+ABCD”格式的答案。"
                         "请确认是答案清单（如 1.A 2.B 3.C…），而非答题卡照片。")
    return ans


def _enhance_image(data: bytes):
    """图片增强预处理：小图放大 + CLAHE 对比度，提升 OCR 小字识别率。"""
    import cv2
    import numpy as np
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    if max(h, w) < 1600:
        s = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def parse_answers_image_enhanced(data: bytes) -> Dict[int, str]:
    """增强预处理后 OCR（与直接 OCR 互为验证路径）。"""
    import cv2
    img = _enhance_image(data)
    if img is None:
        raise ValueError("图片无法读取。")
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("图片预处理失败。")
    return parse_answers_image(png.tobytes())


def parse_pdf_scan_ocr(data: bytes, dpi: int = 200) -> Dict[int, str]:
    """扫描版 PDF：逐页渲染成图 → OCR（文本 + 空间对齐）→ 解析。"""
    import pymupdf  # PyMuPDF
    doc = pymupdf.open(stream=data, filetype="pdf")
    merged: Dict[int, str] = {}
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
        result = _ocr_result(png)
        if result:
            merged.update(_parse_ocr_result(result))
    doc.close()
    if not merged:
        raise ValueError("PDF 未识别出文字（既无文本层也无 OCR 结果）。")
    return dict(sorted(merged.items()))
