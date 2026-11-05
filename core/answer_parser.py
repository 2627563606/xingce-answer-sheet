# -*- coding: utf-8 -*-
"""
答案文件解析器：把 Excel / Word / PDF / TXT 里的标准答案解析成
{题号: "A/B/C/D"} 字典，供判卷使用。

支持两种常见排版：
  1. 表格/两列式：一列题号、一列选项（含表头"题号/答案"）
  2. 文本/逐条式："1.A 2.B 3.C" 或换行 "1.A" / "1 A" / "1、A"
  3. 区间式："1-5 ABCBA" / "6～10: BACDD"（自动展开）
"""
from __future__ import annotations

import io
import re
from typing import Dict, Tuple

OPTIONS = set("ABCD")


# --------------------------------------------------------------------- #
# 文本模式解析
# --------------------------------------------------------------------- #
def _extract_range_answers(text: str) -> Dict[int, str]:
    """区间式：'1-5 ABCBA' / '6～10: BACDD' / '11--20 CCDDB'（双连字符也支持）。"""
    out: Dict[int, str] = {}
    for m in re.finditer(
        r"(?<!\d)(\d{1,3})\s*[-~～至]{1,2}\s*(\d{1,3})\s*[.．:：]?\s*([A-Da-d]{2,})\b",
        text,
    ):
        start, end = int(m.group(1)), int(m.group(2))
        letters = m.group(3).upper()
        if end < start or end - start + 1 != len(letters):
            continue
        if not all(c in OPTIONS for c in letters):
            continue
        for q, c in zip(range(start, end + 1), letters):
            out.setdefault(q, c)  # 先到先得：同题号重复出现时保留第一次
    return out


def _extract_item_answers(text: str) -> Dict[int, str]:
    """逐条式：'1.A' / '2 B' / '3、C' / '4：D'。"""
    out: Dict[int, str] = {}
    for m in re.finditer(r"(?<!\d)(\d{1,3})\s*[.．、:：)）]?\s*([A-Da-d])\b", text):
        q = int(m.group(1))
        c = m.group(2).upper()
        if 1 <= q <= 999 and c in OPTIONS and q not in out:
            out[q] = c
    return out


def _extract_aligned_answers(text: str) -> Dict[int, str]:
    """对齐式（教辅答案页常见）：题号行与答案行上下对齐。

    例：
      1 2 3 4 5
      A A C B B
      6 7 8 9 10
      B A A D A
    按行配对：数字行与下一纯字母行逐列对齐成 {题号: 选项}。
    保守策略：整段配对数 >= 5 才返回，避免零星误判。
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out: Dict[int, str] = {}
    i = 0
    while i < len(lines) - 1:
        nums = re.findall(r"(?<!\d)\d{1,3}(?!\d)", lines[i])
        letters = re.findall(r"[A-Da-d]", lines[i + 1])
        if (nums and letters and len(nums) == len(letters)
                and not re.search(r"[A-Za-z]", lines[i])
                and not re.search(r"\d", lines[i + 1])):
            for q, c in zip(nums, letters):
                qi = int(q)
                if 1 <= qi <= 999:
                    out.setdefault(qi, c.upper())  # 先到先得
            i += 2
            continue
        i += 1
    return out if len(out) >= 5 else {}


def _drop_difference_section(text: str) -> str:
    """剔除文本中「差异题」段落的答案，防止差异题编号(常从1起)覆盖主卷答案。

    教辅答案 PDF/Word 常在答案后附「差异题/差异题部分」段落（副省/地市/执法差异题），
    其题号通常重新从 1 编号，与主卷题号冲突，且**不一定在文档末尾**。
    检测到差异题**标题行**后进入丢弃模式，丢弃其后的答案行；
    一旦遇到下一个**主卷模块标题**（如“一、政治理论（1--20）”“常识判断（21--35）”，
    特征=模块名+括号题号区间）即恢复保留，从而只剔除差异题段落本身。
    """
    _DIFF_TITLE = re.compile(
        r"^\s*[（(【\[]?\s*差异\s*(?:题\s*部分|部分|题|卷|篇)?\s*[）)】\]]?\s*[:：]?\s*$"
    )
    _SECTION_TITLE = re.compile(
        r"^\s*[一二三四五六七八九十]{0,2}\s*[、.．]?\s*[\u4e00-\u9fa5]{2,8}"
        r"\s*[（(]\s*\d{1,3}\s*[-~～至]{1,2}\s*\d{1,3}\s*[）)]"
    )
    lines = text.splitlines()
    in_diff = False
    changed = False
    kept = []
    for ln in lines:
        s = ln.strip()
        if not in_diff:
            if _DIFF_TITLE.match(s):
                in_diff = True
                changed = True
                kept.append("")  # 差异题标题占位
            else:
                kept.append(ln)
        else:
            if _SECTION_TITLE.match(s):
                in_diff = False
                kept.append(ln)  # 回到主卷模块，恢复保留
            else:
                kept.append("")  # 差异题段内容丢弃
    return "\n".join(kept) if changed else text


def parse_answers_text(text: str) -> Dict[int, str]:
    """综合文本解析：区间式 → 逐条式 → 对齐式，逐层补齐，最后按题号排序。"""
    text = _drop_difference_section(text)  # 先剔除差异题段，避免覆盖主卷
    result: Dict[int, str] = {}
    for q, c in _extract_range_answers(text).items():
        result.setdefault(q, c)
    for q, c in _extract_item_answers(text).items():
        result.setdefault(q, c)
    for q, c in _extract_aligned_answers(text).items():
        result.setdefault(q, c)
    result = {q: c for q, c in result.items() if c in OPTIONS}
    return dict(sorted(result.items()))


# --------------------------------------------------------------------- #
# 表格/结构化解析
# --------------------------------------------------------------------- #
def _table_structured(df) -> Dict[int, str] | None:
    """结构化表格 → 答案字典；支持三种排版，否则 None。

    1) 两列式：第一列数字题号、第二列 ABCD 选项（含表头）
    2) 多列成对：每行 偶数 列，0/2/4.. 为题号、1/3/5.. 为答案
    3) 两行式：表格仅两行，第一行题号、第二行答案（按列对齐）
    """
    import pandas as pd
    try:
        df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    except Exception:
        return None
    if df.shape[1] < 2:
        return None
    head_words = ["题号", "题", "序号", "答案", "选项", "正确答案", "标准答案", "选择"]
    while len(df) and any(
        any(w in str(c).strip() for w in head_words) for c in df.iloc[0]
    ):
        df = df.iloc[1:].reset_index(drop=True)
    if not len(df):
        return None
    out: Dict[int, str] = {}

    # 3) 两行式：第一行全数字=题号、第二行全字母=答案
    if len(df) == 2:
        try:
            r0 = [int(float(str(v).strip())) for v in df.iloc[0]]
        except (ValueError, TypeError):
            r0 = []
        r1 = [str(v).strip().upper() for v in df.iloc[1]]
        letters = [v for v in r1 if v in OPTIONS]
        if r0 and len(r0) == len(letters) and all(1 <= q <= 999 for q in r0):
            for q, c in zip(r0, letters):
                out.setdefault(q, c)  # 先到先得
            return out

    # 2) 多列成对：偶数列，0/2/4.. 题号、1/3/5.. 答案
    n = df.shape[1]
    if n >= 4 and n % 2 == 0:
        for qcol in range(0, n, 2):
            acol = qcol + 1
            for _, r in df.iterrows():
                try:
                    q = int(float(r.iloc[qcol]))
                except (TypeError, ValueError):
                    continue
                s = str(r.iloc[acol]).strip().upper()
                if 1 <= q <= 999 and s in OPTIONS:
                    out.setdefault(q, s)  # 先到先得
        if out:
            return out

    # 1) 两列式（原有）
    col0, col1 = df.iloc[:, 0], df.iloc[:, 1]
    q_nums, ok = [], True
    for v in col0:
        try:
            q_nums.append(int(float(v)))
        except (TypeError, ValueError):
            ok = False
            break
        if q_nums[-1] < 1 or q_nums[-1] > 999:
            ok = False
            break
    if not ok:
        return None
    for q, v in zip(q_nums, col1):
        s = str(v).strip().upper()
        if s in OPTIONS:
            out.setdefault(q, s)  # 先到先得
    return out if out else None


def _frame_to_text(df) -> str:
    """把表格展平为逐行文本（供文本模式解析）。"""
    import pandas as pd
    lines = []
    for _, row in df.iterrows():
        cells = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
        if cells:
            lines.append(" ".join(cells))
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# 各文件格式入口
# --------------------------------------------------------------------- #
def parse_excel(data: bytes) -> Dict[int, str]:
    import pandas as pd
    df = pd.read_excel(io.BytesIO(data), header=None)
    if df.empty:
        raise ValueError("Excel 内容为空")
    structured = _table_structured(df)
    if structured:
        return structured
    return parse_answers_text(_frame_to_text(df))


def parse_word(data: bytes) -> Dict[int, str]:
    from docx import Document
    import pandas as pd
    doc = Document(io.BytesIO(data))
    merged = parse_answers_text("\n".join(p.text for p in doc.paragraphs))
    for table in doc.tables:
        tbl = [[c.text for c in row.cells] for row in table.rows]
        out = _table_structured(pd.DataFrame(tbl))
        if out:
            for q, c in out.items():
                merged.setdefault(q, c)  # 先到先得：后续表格不覆盖已有答案
        else:
            for row in tbl:
                for q, c in parse_answers_text(" ".join(row)).items():
                    merged.setdefault(q, c)
    return dict(sorted(merged.items()))


def parse_pdf(data: bytes) -> Dict[int, str]:
    """PDF 解析：优先提取文字层；文字太少（如扫描版）时回退到 OCR。"""
    txt_ans = _pdf_text_answers(data)
    if len(txt_ans) >= 10:
        return txt_ans
    from .answer_ocr import parse_pdf_scan_ocr
    return parse_pdf_scan_ocr(data)


def _pdf_text_answers(data: bytes) -> Dict[int, str]:
    """仅文字层：pdfplumber 逐页提取文本/表格 → 解析；跨页冲突时**靠前页优先**。

    （教辅答案 PDF 常含“差异题”页，同一题号答案不同；主卷在前，应优先保留。）
    """
    import pdfplumber
    out: Dict[int, str] = {}
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                for q, c in parse_answers_text(t).items():
                    out.setdefault(q, c)
            for tbl in page.extract_tables() or []:
                import pandas as pd
                for q, c in (_table_structured(pd.DataFrame(tbl)) or {}).items():
                    out.setdefault(q, c)
    return out


def _pdf_pages_text_answers(data: bytes) -> list:
    """逐页提取文字层答案，返回 [(page_idx, {题号:选项}), ...]（每页独立）。

    用于多识别方式：同一 PDF 的每一页作为一个独立识别方式，
    跨页答案不一致（如差异题页）时由上层冲突检测交给用户判断。
    """
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for idx, page in enumerate(pdf.pages):
            t = page.extract_text()
            ans: Dict[int, str] = {}
            if t:
                ans.update(parse_answers_text(t))
            for tbl in page.extract_tables() or []:
                import pandas as pd
                ans.update(_table_structured(pd.DataFrame(tbl)) or {})
            if ans:
                pages.append((idx, dict(sorted(ans.items()))))
    return pages


def parse_answers_file_multi(filename: str, data: bytes) -> list:
    """对**同一份答案文件**用多种识别/解析方式做交叉验证。

    返回 [{"name": 方式名, "answers": {题号:选项}}, ...]（仅含非空结果）。
    同一文件可能给出多条解析路径（如 PDF：文字层提取 vs OCR 识别），
    由上层对比、冲突时交用户判断。
    """
    name = (filename or "").lower()
    paths: list = []
    if name.endswith((".xlsx", ".xls")):
        # 结构化表格 vs 文本展平
        import pandas as pd
        try:
            df = pd.read_excel(io.BytesIO(data), header=None)
            st = _table_structured(df)
            if st:
                paths.append({"name": "结构化表格", "answers": st})
            tx = parse_answers_text(_frame_to_text(df))
            if tx:
                paths.append({"name": "文本模式", "answers": tx})
        except Exception:
            pass
    elif name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            merged = parse_answers_text("\n".join(p.text for p in doc.paragraphs))
            if merged:
                paths.append({"name": "段落文本", "answers": merged})
            for table in doc.tables:
                import pandas as pd
                out = _table_structured(pd.DataFrame([[c.text for c in r.cells]
                                                      for r in table.rows]))
                if out:
                    paths.append({"name": "表格", "answers": out})
        except Exception:
            pass
    elif name.endswith(".pdf"):
        # 每页文字层作为独立识别方式（差异题页与主卷页冲突时交用户判断）
        try:
            for pidx, p_ans in _pdf_pages_text_answers(data):
                paths.append({"name": f"第{pidx+1}页文字", "answers": p_ans})
        except Exception:
            pass
        try:
            from .answer_ocr import parse_pdf_scan_ocr
            ocr_ans = parse_pdf_scan_ocr(data)
            if ocr_ans:
                paths.append({"name": "OCR识别", "answers": ocr_ans})
        except Exception:
            pass
    elif name.endswith((".txt", ".csv")):
        text = data.decode("utf-8", errors="ignore")
        ans = parse_answers_text(text)
        if ans:
            paths.append({"name": "文本解析", "answers": ans})
    elif name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        # 直接 OCR vs 增强预处理 OCR（两种识别方式互为验证）
        from .answer_ocr import parse_answers_image, parse_answers_image_enhanced
        try:
            a = parse_answers_image(data)
            if a:
                paths.append({"name": "直接OCR", "answers": a})
        except Exception:
            pass
        try:
            a = parse_answers_image_enhanced(data)
            if a:
                paths.append({"name": "增强OCR", "answers": a})
        except Exception:
            pass
    return paths


def parse_answers_file(filename: str, data: bytes) -> Tuple[Dict[int, str], str]:
    """统一入口：返回 (标准答案字典, 来源格式说明)。

    支持：Excel / Word / PDF / TXT / CSV，以及拍照/扫描的答案图片（jpg/png，
    走 OCR）。PDF 为扫描版时自动 OCR。
    """
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return parse_excel(data), "Excel"
    if name.endswith(".docx"):
        return parse_word(data), "Word"
    if name.endswith(".pdf"):
        return parse_pdf(data), "PDF"
    if name.endswith((".txt", ".csv")):
        text = data.decode("utf-8", errors="ignore")
        return parse_answers_text(text), "文本"
    if name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        from .answer_ocr import parse_answers_image
        return parse_answers_image(data), "图片OCR"
    raise ValueError("不支持的文件格式，请上传 Excel(.xlsx/.xls) / Word(.docx) / PDF / TXT / 图片(jpg/png)")
