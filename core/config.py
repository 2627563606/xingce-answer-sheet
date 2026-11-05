# -*- coding: utf-8 -*-
"""
配置加载模块
- 标准答案配置文件 (standard_answers.json)
- 模块分值配置文件 (module_weights.json)
均支持用户自定义，若文件缺失/损坏则回退到内置默认值。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Tuple

# 项目根目录（core 的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DB_PATH = os.path.join(BASE_DIR, "records.db")

DEFAULT_STANDARD_ANSWERS_PATH = os.path.join(CONFIG_DIR, "standard_answers.json")
DEFAULT_MODULE_WEIGHTS_PATH = os.path.join(CONFIG_DIR, "module_weights.json")
CURRENT_USER_PATH = os.path.join(CONFIG_DIR, "current_user.json")

# 行测五大模块默认配置（题目数 130）
DEFAULT_MODULES: Dict[str, Dict] = {
    "常识判断": {"start": 1, "end": 20, "weight": 0.6},
    "言语理解": {"start": 21, "end": 50, "weight": 0.8},
    "数量关系": {"start": 51, "end": 65, "weight": 1.0},
    "判断推理": {"start": 66, "end": 100, "weight": 0.7},
    "资料分析": {"start": 101, "end": 130, "weight": 0.9},
}

OPTIONS = ["A", "B", "C", "D"]


def load_current_user() -> str:
    """读取上次使用的当前用户（用于自动带出，避免每次重填）。"""
    try:
        with open(CURRENT_USER_PATH, encoding="utf-8") as f:
            return str(json.load(f).get("user", "")).strip()
    except Exception:
        return ""


def save_current_user(user: str) -> None:
    """记住当前用户，下次打开系统自动带出。"""
    try:
        with open(CURRENT_USER_PATH, "w", encoding="utf-8") as f:
            json.dump({"user": user}, f, ensure_ascii=False)
    except OSError:
        pass


def load_json(path: str, default: dict) -> dict:
    """读取 JSON 配置文件，失败时回退到 default。"""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and len(data) > 0:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return default


def load_standard_answers(user: str = "", path: str | None = None) -> Dict[str, str]:
    """加载标准答案：{"题号": "A/B/C/D"}，键统一为字符串。

    user 非空时读取该用户工作区（新用户首次回退到全局模板，保存后隔离）。
    """
    p = _user_ans_path(user, path)
    if user and not os.path.exists(p):
        p = DEFAULT_STANDARD_ANSWERS_PATH
    raw = load_json(p, {})
    answers: Dict[str, str] = {}
    for k, v in raw.items():
        s = str(v).strip().upper()
        if s in OPTIONS:
            answers[str(k)] = s
    return answers


def load_module_weights(user: str = "", path: str | None = None) -> Dict[str, Dict]:
    """加载模块分值配置并做基本校验，返回 {模块名: {start, end, weight}}。

    user 非空时读取该用户工作区（新用户首次回退到全局模板，保存后隔离）。
    """
    p = _user_mod_path(user, path)
    if user and not os.path.exists(p):
        p = DEFAULT_MODULE_WEIGHTS_PATH
    raw = load_json(p, DEFAULT_MODULES)
    modules: Dict[str, Dict] = {}
    for name, cfg in raw.items():
        try:
            start = int(cfg.get("start"))
            end = int(cfg.get("end"))
            weight = float(cfg.get("weight"))
            if start >= 1 and end >= start and weight > 0:
                modules[str(name)] = {"start": start, "end": end, "weight": weight}
        except (TypeError, ValueError, AttributeError):
            continue
    if not modules:
        modules = deepcopy(DEFAULT_MODULES)
    return modules


def total_questions(modules: Dict[str, Dict]) -> int:
    """由模块配置推导总题数（最大 end）。"""
    return max((m["end"] for m in modules.values()), default=130)


def adapt_modules_to_total(modules: Dict[str, Dict], total) -> Dict[str, Dict]:
    """把模块配置适配到指定总题量（不同地区/试卷类型题量不同）。

    - 若 total ≤ 所有模块最大 end：原样返回；
    - 若 total > 最大 end：把 end 最大的模块自动扩展到 total，
      使超出的题（如 131-140 题）按该模块权重正常计分，而不是不计分。
    返回新字典，不改动传入对象。
    """
    if not modules or total is None:
        return modules
    try:
        total = int(total)
    except (TypeError, ValueError):
        return modules
    if total < 1:
        return modules
    max_end = max(m["end"] for m in modules.values())
    if total <= max_end:
        return modules
    last_name = max(modules, key=lambda n: modules[n]["end"])
    adapted = {n: dict(m) for n, m in modules.items()}
    adapted[last_name]["end"] = total
    return adapted


def question_to_module(modules: Dict[str, Dict]) -> Dict[int, Tuple[str, float]]:
    """返回 {题号: (模块名, 权重)} 映射，题号不在任何模块时权重为 0。"""
    mapping: Dict[int, Tuple[str, float]] = {}
    for name, cfg in modules.items():
        for q in range(cfg["start"], cfg["end"] + 1):
            mapping[q] = (name, cfg["weight"])
    return mapping


def save_standard_answers(answers: Dict[str, str], user: str = "", path: str | None = None) -> str:
    """将标准答案写回 JSON（user 非空 → 写入该用户工作区）。"""
    p = _user_ans_path(user, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    return p


def save_module_weights(modules: Dict[str, Dict], user: str = "", path: str | None = None) -> str:
    """将模块配置写回 JSON（user 非空 → 写入该用户工作区）。"""
    p = _user_mod_path(user, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(modules, f, ensure_ascii=False, indent=2)
    return p


# --------------------------------------------------------------------------- #
# 用户工作区（多用户完全隔离）：每个用户一份配置/答案/密码，历史按 owner 隔离
# --------------------------------------------------------------------------- #
def _safe_username(user: str) -> str:
    """用户名 → 安全目录名（过滤路径分隔符，防路径穿越）。"""
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(user).strip())
    return s or "_default"


def _user_dir(user: str) -> str:
    return os.path.join(CONFIG_DIR, "users", _safe_username(user))


def user_workspace(user: str) -> str:
    """用户工作区目录；user 为空 → 全局 configs（匿名）。"""
    if not user:
        return CONFIG_DIR
    d = _user_dir(user)
    os.makedirs(d, exist_ok=True)
    return d


def _user_ans_path(user: str, path: str | None) -> str:
    return path or os.path.join(user_workspace(user), "standard_answers.json")


def _user_mod_path(user: str, path: str | None) -> str:
    return path or os.path.join(user_workspace(user), "module_weights.json")


def user_exists(user: str) -> bool:
    """该用户是否已建立过工作区（有配置/答案/密码任一文件）。"""
    u = str(user or "").strip()
    if not u:
        return False
    d = _user_dir(u)
    return os.path.isdir(d) and any(
        os.path.exists(os.path.join(d, f))
        for f in ("auth.json", "standard_answers.json", "module_weights.json"))


def user_has_password(user: str) -> bool:
    """该用户是否设置了密码（未设置 → 允许本机自动恢复登录）。"""
    u = str(user or "").strip()
    if not u:
        return False
    p = os.path.join(_user_dir(u), "auth.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return bool(data.get("hash"))


def user_set_password(user: str, password: str) -> None:
    """首次进入时设置密码（留空=不设密码）。"""
    u = str(user or "").strip()
    if not u:
        return
    d = _user_dir(u)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "auth.json"), "w", encoding="utf-8") as f:
        json.dump({"hash": _pwd_hash(password)}, f, ensure_ascii=False)


def user_check_password(user: str, password: str) -> bool:
    """校验密码；该用户未设置密码 → 直接通过。"""
    u = str(user or "").strip()
    if not u:
        return True
    p = os.path.join(_user_dir(u), "auth.json")
    if not os.path.exists(p):
        return True
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return True
    h = data.get("hash", "")
    if not h:
        return True
    return _pwd_hash(password) == h


def _pwd_hash(pwd: str) -> str:
    return hashlib.sha256(("xingce::" + str(pwd)).encode("utf-8")).hexdigest()
