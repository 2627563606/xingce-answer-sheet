# -*- coding: utf-8 -*-
"""
历史记录与统计分析（SQLite）

字段：id, 考生姓名, 准考证号, 卷别, 总分, 各模块得分(JSON),
      各模块正确率(JSON), 错题详情(JSON), 识别结果(JSON), 批次, 时间戳
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Dict, List, Optional

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    owner             TEXT DEFAULT '',
    name              TEXT DEFAULT '',
    registration      TEXT DEFAULT '',
    paper_code        TEXT DEFAULT '',
    total_score       REAL DEFAULT 0,
    module_scores     TEXT DEFAULT '{}',
    module_accuracy   TEXT DEFAULT '{}',
    wrong_details     TEXT DEFAULT '[]',
    recognized_answers TEXT DEFAULT '{}',
    batch_id          TEXT DEFAULT '',
    created_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_reg ON records(registration);
CREATE INDEX IF NOT EXISTS idx_records_name ON records(name);
CREATE INDEX IF NOT EXISTS idx_records_time ON records(created_at);
"""


class RecordDB:
    def __init__(self, db_path: str = DB_PATH):
        self.path = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)
            # 兼容旧库：补齐 owner 列（老记录 owner 为空，视为不隔离/公共），
            # 索引必须在本列补齐之后再建，否则旧库无列会报错。
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(records)")}
            if "owner" not in cols:
                conn.execute("ALTER TABLE records ADD COLUMN owner TEXT DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_owner ON records(owner)")

    # ------------------------------------------------------------ #
    def insert(
        self,
        owner: str = "",
        name: str = "",
        registration: str = "",
        paper_code: str = "",
        total_score: float = 0.0,
        module_scores: Optional[Dict] = None,
        module_accuracy: Optional[Dict] = None,
        wrong_details: Optional[List] = None,
        recognized_answers: Optional[Dict] = None,
        batch_id: str = "",
    ) -> int:
        with closing(self._conn()) as conn, conn:
            cur = conn.execute(
                """INSERT INTO records
                   (owner, name, registration, paper_code, total_score, module_scores,
                    module_accuracy, wrong_details, recognized_answers, batch_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    owner or "",
                    name or "",
                    registration or "",
                    paper_code or "",
                    round(float(total_score), 2),
                    json.dumps(module_scores or {}, ensure_ascii=False),
                    json.dumps(module_accuracy or {}, ensure_ascii=False),
                    json.dumps(wrong_details or [], ensure_ascii=False),
                    json.dumps(recognized_answers or {}, ensure_ascii=False),
                    batch_id or "",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            return int(cur.lastrowid)

    def list_records(self, search: str = "", owner: str = "", limit: int = 500) -> List[Dict]:
        """按时间倒序返回记录；owner 非空时只返回该用户记录；search 匹配姓名/准考证号。"""
        q = "SELECT * FROM records"
        params: list = []
        conds: list = []
        if owner:
            conds.append("owner=?")
            params.append(owner)
        if search:
            conds.append("(name LIKE ? OR registration LIKE ?)")
            params += [f"%{search}%", f"%{search}%"]
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with closing(self._conn()) as conn, conn:
            rows = conn.execute(q, tuple(params)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, rid: int) -> Optional[Dict]:
        with closing(self._conn()) as conn, conn:
            row = conn.execute("SELECT * FROM records WHERE id=?", (rid,)).fetchone()
        return self._row_to_dict(row) if row else None

    def delete(self, rid: int, owner: str = "") -> bool:
        """删除记录。owner 非空时只允许删除属于该用户的记录。"""
        with closing(self._conn()) as conn, conn:
            if owner:
                cur = conn.execute(
                    "DELETE FROM records WHERE id=? AND owner=?", (rid, owner))
            else:
                cur = conn.execute("DELETE FROM records WHERE id=?", (rid,))
        return cur.rowcount > 0

    def delete_batch(self, batch_id: str, owner: str = "") -> int:
        with closing(self._conn()) as conn, conn:
            if owner:
                cur = conn.execute(
                    "DELETE FROM records WHERE batch_id=? AND owner=?", (batch_id, owner))
            else:
                cur = conn.execute("DELETE FROM records WHERE batch_id=?", (batch_id,))
        return cur.rowcount

    def candidate_records(self, name: str = "", registration: str = "", owner: str = "") -> List[Dict]:
        """按考生（姓名或准考证号）检索其历次记录，用于趋势图；owner 非空时限定用户。"""
        q = "SELECT * FROM records WHERE 1=1"
        params: list = []
        if owner:
            q += " AND owner=?"
            params.append(owner)
        if registration:
            q += " AND registration=?"
            params.append(registration)
        elif name:
            q += " AND name=?"
            params.append(name)
        q += " ORDER BY created_at ASC, id ASC"
        with closing(self._conn()) as conn, conn:
            rows = conn.execute(q, tuple(params)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def wrong_frequency(self, qmax: int = 200, owner: str = "") -> Dict[int, int]:
        """统计错题频次 {题号: 次数}；owner 非空时只统计该用户。"""
        freq: Dict[int, int] = {}
        with closing(self._conn()) as conn, conn:
            if owner:
                rows = conn.execute(
                    "SELECT wrong_details FROM records WHERE owner=?", (owner,)).fetchall()
            else:
                rows = conn.execute("SELECT wrong_details FROM records").fetchall()
        for r in rows:
            try:
                details = json.loads(r["wrong_details"])
            except (json.JSONDecodeError, TypeError):
                continue
            for d in details:
                try:
                    qno = int(d.get("题号", d.get("qno", 0)))
                except (ValueError, TypeError):
                    continue
                if 1 <= qno <= qmax:
                    freq[qno] = freq.get(qno, 0) + 1
        return freq

    def module_average_accuracy(self, owner: str = "") -> Dict[str, float]:
        """各模块正确率均值；owner 非空时只统计该用户。"""
        acc: Dict[str, List[float]] = {}
        with closing(self._conn()) as conn, conn:
            if owner:
                rows = conn.execute(
                    "SELECT module_accuracy FROM records WHERE owner=?", (owner,)).fetchall()
            else:
                rows = conn.execute("SELECT module_accuracy FROM records").fetchall()
        for r in rows:
            try:
                data = json.loads(r["module_accuracy"])
            except (json.JSONDecodeError, TypeError):
                continue
            for k, v in data.items():
                try:
                    acc.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    continue
        return {k: round(sum(v) / len(v), 4) for k, v in acc.items() if v}

    def stats_summary(self, owner: str = "") -> Dict:
        """概览：记录数、平均分、最高分；owner 非空时只统计该用户。"""
        with closing(self._conn()) as conn, conn:
            if owner:
                row = conn.execute(
                    "SELECT COUNT(*) c, AVG(total_score) avg, MAX(total_score) mx "
                    "FROM records WHERE owner=?", (owner,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) c, AVG(total_score) avg, MAX(total_score) mx "
                    "FROM records").fetchone()
        return {
            "count": row["c"] or 0,
            "avg_score": round(row["avg"] or 0, 2),
            "max_score": round(row["mx"] or 0, 2),
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        d = dict(row)
        for k in ("module_scores", "module_accuracy", "wrong_details", "recognized_answers"):
            try:
                d[k] = json.loads(d[k]) if d[k] else ({} if k != "wrong_details" else [])
            except (json.JSONDecodeError, TypeError):
                d[k] = [] if k == "wrong_details" else {}
        return d
