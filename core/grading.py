# -*- coding: utf-8 -*-
"""
判卷与计分模块（行测模块化配置）

计分规则：
  每题得分 = 该模块权重（单题分值）
  normalize_to_100=True 时按「模块权重之和」归一化到 100 分：
      module_score  = 该模块答对题数 × 权重 / 总权重 × 100
      total_score   = Σ module_score
  模块正确率 = 该模块答对题数 / 该模块总题数
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import adapt_modules_to_total, question_to_module


@dataclass
class GradingResult:
    total_score: float = 0.0
    total_weight: float = 0.0
    raw_total: float = 0.0
    module_scores: Dict[str, float] = field(default_factory=dict)      # 模块得分（归一化后）
    module_raw: Dict[str, float] = field(default_factory=dict)         # 模块原始加权分
    module_accuracy: Dict[str, float] = field(default_factory=dict)    # 模块正确率 0~1
    module_correct: Dict[str, int] = field(default_factory=dict)
    module_total: Dict[str, int] = field(default_factory=dict)
    correct_count: int = 0
    total_count: int = 0
    answered_count: int = 0
    blank_count: int = 0
    wrong_details: List[Dict] = field(default_factory=list)   # [{题号, 识别, 标准}]
    detail: Dict[int, Dict] = field(default_factory=dict)    # {题号: {rec, std, correct}}


def grade(
    recognized: Dict[int, str],
    standard: Dict[str, str],
    modules: Dict[str, Dict],
    normalize_to_100: bool = True,
    total_questions: Optional[int] = None,
) -> GradingResult:
    """判卷。

    recognized: {题号(int): 选项}
    standard:   {题号(str): 选项}   —— 支持 int/str 混合键
    modules:    {模块名: {start, end, weight}}
    total_questions: 本卷实际总题量（不同地区/试卷题量不同）。传入后若超出
        模块配置的最大 end，会自动扩展最后一个模块覆盖到该题量，保证超出的
        题也参与计分。None 时按模块配置原样（只统计配置范围内）。
    """
    res = GradingResult()
    if total_questions:
        modules = adapt_modules_to_total(modules, total_questions)
    mapping = question_to_module(modules)          # {题号: (模块名, 权重)}
    std: Dict[int, str] = {}
    for k, v in standard.items():
        try:
            std[int(k)] = v.upper()
        except (ValueError, TypeError):
            continue

    all_q = set(mapping.keys()) | set(recognized.keys()) | set(std.keys())
    if not all_q:
        return res

    # 归一化分母：Σ(该模块题数 × 单题权重)，即按“每题得分=模块权重”加总
    total_weight = sum(w for _name, w in mapping.values())
    res.total_weight = round(total_weight, 4)

    # 每个模块：答对数 / 题数 / 原始加权分
    mod_correct: Dict[str, int] = {}
    mod_total: Dict[str, int] = {}
    mod_raw: Dict[str, float] = {}
    for name, m in modules.items():
        mod_correct[name] = 0
        mod_total[name] = m["end"] - m["start"] + 1
        mod_raw[name] = 0.0

    detail: Dict[int, Dict] = {}
    wrong_details: List[Dict] = []
    raw_total = 0.0
    correct_count = 0
    blank_count = 0

    for q in sorted(all_q):
        rec = recognized.get(q)
        std_ans = std.get(q)
        correct = bool(rec and std_ans and rec == std_ans)

        if rec is None or rec == "":
            blank_count += 1
        elif std_ans is not None:
            if correct:
                correct_count += 1
            else:
                wrong_details.append({
                    "题号": q,
                    "识别": rec,
                    "标准": std_ans,
                })

        detail[q] = {"rec": rec or "", "std": std_ans or "", "correct": correct}

        name, weight = mapping.get(q, ("", 0.0))
        if correct and name:
            mod_correct[name] += 1
            mod_raw[name] += weight
            raw_total += weight

    # 归一化到 100 分
    res.raw_total = round(raw_total, 4)
    if normalize_to_100 and total_weight > 0:
        res.total_score = round(raw_total / total_weight * 100.0, 4)
        for name in modules:
            res.module_scores[name] = round(mod_raw[name] / total_weight * 100.0, 4)
    else:
        res.total_score = round(raw_total, 4)
        res.module_scores = {name: round(mod_raw[name], 4) for name in modules}

    res.module_raw = {k: round(v, 4) for k, v in mod_raw.items()}
    res.module_correct = mod_correct
    res.module_total = mod_total
    res.module_accuracy = {
        name: round(mod_correct[name] / mod_total[name], 6) if mod_total[name] else 0.0
        for name in modules
    }
    res.correct_count = correct_count
    res.total_count = len(all_q)
    res.answered_count = res.total_count - blank_count
    res.blank_count = blank_count
    res.wrong_details = wrong_details
    res.detail = detail
    return res


def answers_from_details(detail: Dict[int, Dict]) -> Dict[int, str]:
    """从判定详情还原识别答案（供界面手动修正后重新计分）。"""
    return {q: d["rec"] for q, d in detail.items() if d.get("rec")}
