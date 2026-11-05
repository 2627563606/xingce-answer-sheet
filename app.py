# -*- coding: utf-8 -*-
"""
行测答题卡扫描评分系统 —— Streamlit 界面

启动： streamlit run app.py
流程：
  1. 上传答题卡照片 → 「识别并批改」
  2. 右侧展示识别结果（绿色=正确 / 红色=错误，错题标红），可手动修正识别结果
  3. 展示总分 + 五大模块得分柱状图 + 模块正确率 + 雷达图
  4. 填写考生姓名（自动识别准考证号）→ 保存记录入 SQLite 历史库
  附加：批量导入 / 历史记录与趋势 / 模块雷达图 / 错题热力图 / 配置管理
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import re
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

# 服务器资源受限（1.6G 内存 / 2 核），限制 OpenCV 线程数避免识别时 CPU 打满拖垮其他项目
cv2.setNumThreads(2)

from core.answer_parser import parse_answers_file, parse_answers_file_multi
from core.answer_sheet import AnswerSheetRecognizer
from core.config import (
    OPTIONS,
    load_module_weights,
    load_standard_answers,
    question_to_module,
    save_module_weights,
    save_standard_answers,
    user_check_password,
    user_exists,
    user_has_password,
    user_set_password,
)
from core.database import RecordDB
from core.grading import grade

st.set_page_config(page_title="行测答题卡扫描评分系统", layout="wide", page_icon="📝")

# --------------------------------------------------------------------- #
# 全局工具
# --------------------------------------------------------------------- #
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image(uploaded, target_edge: int = 1700) -> np.ndarray | None:
    """读取上传图片并**统一归一化**（后端等效"微信压缩"处理），保证跨设备一致。

    识别算法对输入分辨率/方向/格式极敏感（实测 1600-1850px 为稳定正确档，2000px 错位）。
    手机原图带 EXIF 方向标记（像素横置、相册显示正）、微信转发已实际转正、浏览器可能转 PNG——
    各种输入格式/方向不同会导致同一张图判分不一致。这里统一为「正向纵向 + 1700px」：
      1) PIL 读入并 exif_transpose：按 EXIF 方向标记转正像素（处理手机原图/微信压缩差异）
      2) 方向归一化：答题卡为竖长版式，横置(宽>高)一律转正为纵向（兜底 PNG 丢失方向信息）
      3) 统一缩放到最长边 target_edge
    """
    try:
        uploaded.seek(0)
        pil_img = Image.open(uploaded)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # PIL 打不开（罕见格式）时退回 OpenCV 直接解码
        data = np.frombuffer(uploaded.getvalue(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
    # 方向归一化：行测答题卡为竖长版式，横置(宽>高)视为拍摄/格式方向异常，转正为纵向
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    longest = max(h, w)
    if abs(longest - target_edge) > 30:
        scale = target_edge / longest
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
    return img


def load_configs(user: str = ""):
    # 注意：不能加 @st.cache_data —— 用户会在「配置管理」保存模块配置/标准答案，
    # 若缓存则 JSON 更新后刷新仍返回旧值（表现为“修改后没了”）。
    # 配置为低频小数据，每次直接读 JSON 即可。
    return load_standard_answers(user=user), load_module_weights(user=user)


def current_user() -> str:
    """当前登录用户；未登录返回空（匿名/全局）。"""
    if st.session_state.get("logged_in"):
        return str(st.session_state.get("current_user", "")).strip()
    return ""


def answers_df(answers, standard) -> pd.DataFrame:
    """识别结果 → 表格（供手动修正）。"""
    rows = []
    std_keys = {int(k) for k in standard.keys()}
    all_q = sorted(set(answers.keys()) | std_keys)
    for q in all_q:
        rec = answers.get(q, "")
        std = standard.get(str(q), standard.get(q, ""))
        rows.append({
            "题号": q,
            "识别选项": rec,
            "标准答案": std,
            "判定": "✓" if (rec and rec == std) else ("✗" if rec else "未涂"),
        })
    return pd.DataFrame(rows)


def df_to_answers(df: pd.DataFrame) -> dict:
    """表格 → {题号: 选项}（去除空白行）。"""
    out = {}
    for _, r in df.iterrows():
        q = int(r["题号"])
        v = str(r["识别选项"]).strip().upper()
        if v in OPTIONS:
            out[q] = v
    return out


def order_modules(known_modules: list) -> list:
    """把模块列表重排为「配置管理」中的填写顺序；未配置的模块追加到末尾。

    所有图表（柱状图/雷达图/趋势图）统一用此顺序，保证全系统一致。
    """
    cfg = list((st.session_state.get("modules") or {}).keys())
    if not cfg:
        return list(known_modules)
    seen = set()
    out = []
    for m in cfg:
        if m in known_modules and m not in seen:
            out.append(m)
            seen.add(m)
    for m in known_modules:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def recalc_from_record(record: dict, modules: dict):
    """从历史记录的原始答题数据按「当前模块配置」重新判分。

    重建 recognized（用户答案）和 standard（正确答案）：
      - 用户答案从 recognized_answers 字段取
      - 正确答案：错题从 wrong_details 的「标准」取，答对的题标准=用户答案
    成功返回 GradingResult；记录缺少原始答题数据时返回 None（调用方回退到存储分数）。
    """
    from core.grading import grade
    recognized_raw = record.get("recognized_answers") or {}
    if not recognized_raw:
        return None
    recognized = {}
    for k, v in recognized_raw.items():
        try:
            recognized[int(k)] = str(v).upper()
        except (ValueError, TypeError):
            continue
    wrong = record.get("wrong_details") or []
    standard = {}
    wrong_q = set()
    for d in wrong:
        q = d.get("题号")
        std = d.get("标准")
        if q is not None and std:
            standard[int(q)] = str(std).upper()
            wrong_q.add(int(q))
    for q, rec in recognized.items():
        if q not in wrong_q and rec:
            standard[q] = rec
    if not standard:
        return None
    all_q = set(recognized.keys()) | set(standard.keys())
    total_q = max(all_q) if all_q else None
    return grade(recognized, standard, modules, total_questions=total_q)


def module_bar(grad) -> go.Figure:
    mods = order_modules(list(grad.module_scores.keys()))
    scores = [grad.module_scores[m] for m in mods]
    acc = [grad.module_accuracy[m] * 100 for m in mods]
    fig = go.Figure(go.Bar(
        x=mods, y=scores, name="模块得分",
        text=[f"{s:.2f} 分<br>{a:.2f}% 正确" for s, a in zip(scores, acc)],
        textposition="outside",
        marker_color=px.colors.qualitative.Set2,
        width=0.6,  # 统一所有柱子宽度，避免宽窄不一
    ))
    fig.update_layout(title="各模块得分", yaxis_title="模块得分(每题权重)",
                      height=380, showlegend=False)
    return fig


def radar_chart(grad) -> go.Figure:
    mods = order_modules(list(grad.module_accuracy.keys()))
    vals = [grad.module_accuracy[m] * 100 for m in mods]
    fig = go.Figure(go.Scatterpolar(
        r=vals + vals[:1], theta=mods + mods[:1], fill="toself",
        line_color="#e07b39", name="正确率%",
        hovertemplate="%{theta}: %{r:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="各模块正确率雷达图", polar=dict(radialaxis=dict(range=[0, 100])),
                      height=380, showlegend=False)
    return fig


def wrong_heatmap(freq: dict, qmax: int = 130) -> go.Figure:
    xs = list(range(1, qmax + 1))
    z = [freq.get(q, 0) for q in xs]
    fig = go.Figure(go.Heatmap(
        x=xs, y=["错题次数"], z=[z], colorscale="Reds",
        text=[[f"Q{q}:{freq.get(q,0)}" for q in xs]],
        texttemplate="%{text}", textfont={"size": 9},
        colorbar={"title": "次数"},
    ))
    fig.update_layout(title="历次错题热力图（高频错题号标红）", height=260,
                      xaxis=dict(dtick=10), yaxis=dict(autorange="reversed"))
    return fig


def _practice_x_labels(records) -> list:
    """生成折线图横轴标签：按时间排序的日期字符串，等间距（category 轴）。
    同一天多次练习自动加 (2)/(3) 区分，避免 category 重叠。"""
    from collections import Counter
    cnt = Counter()
    labels = []
    for r in records:
        d = pd.to_datetime(r["created_at"]).strftime("%Y-%m-%d")
        cnt[d] += 1
        labels.append(d if cnt[d] == 1 else f"{d}({cnt[d]})")
    return labels


def trend_chart(records) -> go.Figure:
    # 时间正序：先完成的考试在左边
    records = sorted(records, key=lambda r: r.get("created_at") or "")
    df = pd.DataFrame([{
        "时间": lbl,
        "总分": r["total_score"],
        "准考证号": r["registration"], "姓名": r["name"],
    } for r, lbl in zip(records, _practice_x_labels(records))])
    # 纵轴自适应：顶部留 20% 边距容纳数值标签，避免上方被裁
    ymin = float(df["总分"].min())
    ymax = float(df["总分"].max())
    pad = (ymax - ymin) * 0.22 if ymax > ymin else max(ymax * 0.15, 5.0)
    yrange = [max(0.0, ymin - pad * 0.3), ymax + pad]
    fig = px.line(df, x="时间", y="总分", markers=True, text="总分",
                  title="该考生历次行测总分趋势")
    fig.update_traces(textposition="top center",
                      hovertemplate="%{y:.2f} 分<br>%{x}<extra></extra>")
    fig.update_layout(height=380, xaxis=dict(type="category"),
                      yaxis=dict(range=yrange))
    return fig


def module_trend_chart(records) -> go.Figure:
    """各模块得分随历次考试的变化趋势（每条线一个模块，按配置顺序）。"""
    if not records:
        return go.Figure()
    mods = order_modules(list({m for r in records for m in (r["module_scores"] or {})}))
    x_labels = _practice_x_labels(records)
    df = pd.DataFrame([{
        "时间": lbl,
        **{m: (r["module_scores"] or {}).get(m, float("nan")) for m in mods},
    } for r, lbl in zip(records, x_labels)])
    fig = go.Figure()
    for m in mods:
        fig.add_trace(go.Scatter(x=df["时间"], y=df[m], mode="lines+markers", name=m,
                                 hovertemplate="%{x}: %{y:.2f} 分<extra></extra>"))
    fig.update_layout(title="各模块得分变化趋势", height=420,
                      yaxis_title="模块得分", hovermode="x unified",
                      xaxis=dict(type="category"))
    return fig


def module_accuracy_trend_chart(records) -> go.Figure:
    """各模块正确率随历次考试的变化趋势（每条线一个模块，按配置顺序）。"""
    if not records:
        return go.Figure()
    mods = order_modules(list({m for r in records for m in (r["module_accuracy"] or {})}))
    x_labels = _practice_x_labels(records)

    def _pct(r, m):
        v = (r.get("module_accuracy") or {}).get(m)
        try:
            return float(v) * 100
        except (TypeError, ValueError):
            return float("nan")

    df = pd.DataFrame([{
        "时间": lbl,
        **{m: _pct(r, m) for m in mods},
    } for r, lbl in zip(records, x_labels)])
    fig = go.Figure()
    for m in mods:
        fig.add_trace(go.Scatter(x=df["时间"], y=df[m], mode="lines+markers", name=m,
                                 hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="各模块正确率变化趋势", height=420,
                      yaxis_title="正确率(%)", hovermode="x unified",
                      yaxis=dict(range=[0, 100]),
                      xaxis=dict(type="category"))
    return fig


def module_trend_animated_chart(records) -> str:
    """自绘 SVG 多模块折线图：得分 / 正确率 一键切换，requestAnimationFrame 平滑过渡。

    与柱状图同一套动画机制：切换时每条折线的 path、数据点、纵轴刻度同步插值，
    700ms cubic-in-out 缓动，确保丝滑。图例可点击显示/隐藏模块。
    返回 HTML 字符串，调用方用 components.html 渲染。
    """
    if not records:
        return ""
    # 时间正序：先完成的考试在左边
    records = sorted(records, key=lambda r: r.get("created_at") or "")
    mods = order_modules(list(
        {m for r in records for m in (r["module_scores"] or {})}
        | {m for r in records for m in (r["module_accuracy"] or {})}))
    x_labels = _practice_x_labels(records)
    n = len(x_labels)

    def _val(r, m, key, scale=1.0):
        v = (r.get(key) or {}).get(m)
        try:
            return float(v) * scale
        except (TypeError, ValueError):
            return 0.0

    scores = {m: [_val(r, m, "module_scores") for r in records] for m in mods}
    pcts = {m: [_val(r, m, "module_accuracy", 100) for r in records] for m in mods}

    # 得分纵轴范围（显式计算，不用 autorange）
    all_s = [v for m in mods for v in scores[m]]
    s_min, s_max = (min(all_s), max(all_s)) if all_s else (0.0, 10.0)
    pad = (s_max - s_min) * 0.15 if s_max > s_min else max(s_max * 0.1, 1.0)
    score_range = [max(0.0, s_min - pad), s_max + pad]
    pct_range = [0.0, 100.0]

    # 每次考试的详细数据（点击数据点时展示）
    x_labels = _practice_x_labels(records)
    exams = []
    for i, r in enumerate(records):
        ms = r.get("module_scores") or {}
        ma = r.get("module_accuracy") or {}
        exams.append({
            "label": x_labels[i],
            "name": r.get("name") or "",
            "created_at": (r.get("created_at") or "")[:16],
            "total_score": round(float(r.get("total_score", 0)), 2),
            "wrong_count": len(r.get("wrong_details") or []),
            "module_scores": {m: round(float(ms.get(m, 0)), 2) for m in mods},
            "module_accuracy": {m: round(float(ma.get(m, 0)) * 100, 2) for m in mods},
        })

    # SVG 布局
    W, H = 720, 460
    PL, PR, PT, PB = 55, 50, 48, 55
    PW, PH = W - PL - PR, H - PT - PB
    xs = [PL + PW * i / max(n - 1, 1) for i in range(n)]

    def yc(v, ymin, ymax):
        return PT + PH * (1 - (v - ymin) / max(ymax - ymin, 1e-9))

    score_pts = {m: [(xs[i], yc(scores[m][i], *score_range)) for i in range(n)] for m in mods}
    pct_pts = {m: [(xs[i], yc(pcts[m][i], *pct_range)) for i in range(n)] for m in mods}

    def gen_ticks(ymin, ymax):
        return [ymin + (ymax - ymin) * i / 5 for i in range(6)]
    score_ticks = gen_ticks(*score_range)
    pct_ticks = gen_ticks(*pct_range)
    tick_ys = [PT + PH * (1 - i / 5) for i in range(6)]

    colors = px.colors.qualitative.Set2
    mod_colors = {m: colors[i % len(colors)] for i, m in enumerate(mods)}

    def path_d(pts):
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    paths_html, circles_html = [], []
    for mi, m in enumerate(mods):
        c = mod_colors[m]
        paths_html.append(
            f'<path id="mtc_path_{mi}" d="{path_d(score_pts[m])}" '
            f'fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        for i in range(n):
            x, y = score_pts[m][i]
            hover = f"{m}\n{x_labels[i]}\n得分: {scores[m][i]:.2f}\n正确率: {pcts[m][i]:.1f}%\n(点击查看本次考试详情)"
            circles_html.append(
                f'<circle id="mtc_c_{mi}_{i}" cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                f'fill="{c}" stroke="#fff" stroke-width="1.5" '
                f'style="cursor:pointer" onclick="event.stopPropagation(); mtcShowDetail({mi}, {i})">'
                f'<title>{hover}</title></circle>')

    grid_html = []
    for i in range(6):
        ty = tick_ys[i]
        grid_html.append(
            f'<line x1="{PL}" y1="{ty:.1f}" x2="{W-PR}" y2="{ty:.1f}" stroke="#e8e8e8" stroke-width="1"/>')
        grid_html.append(
            f'<text id="mtc_tick_{i}" x="{PL-8}" y="{ty+4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#888">{score_ticks[i]:.1f}</text>')

    xlabels_html = []
    for i in range(n):
        xlabels_html.append(
            f'<text x="{xs[i]:.1f}" y="{H-PB+18}" text-anchor="middle" '
            f'font-size="10" fill="#666">{x_labels[i]}</text>')

    legend_html = []
    for mi, m in enumerate(mods):
        c = mod_colors[m]
        legend_html.append(
            f'<span class="mtc-leg" id="mtc_leg_{mi}" onclick="mtcToggle({mi})" style="--c:{c}">'
            f'<span class="mtc-leg-dot"></span>{m}</span>')

    import json
    js_data = json.dumps({
        "mods": mods, "x_labels": x_labels,
        "score_pts": score_pts, "pct_pts": pct_pts,
        "score_ticks": score_ticks, "pct_ticks": pct_ticks,
        "exams": exams,
    }, ensure_ascii=False)

    html = f"""
    <style>
      .mtc-wrap{{font-family:'Source Sans',sans-serif;background:#fff;border:1px solid #ececec;border-radius:8px;padding:10px 12px;}}
      .mtc-title{{font-size:15px;font-weight:600;color:#333;text-align:center;margin-bottom:6px;}}
      .mtc-btns{{text-align:center;margin-bottom:2px;}}
      .mtc-btns button{{border:1px solid #d0d0d0;background:#fff;color:#444;padding:4px 16px;margin:0 4px;
                        border-radius:16px;cursor:pointer;font-size:13px;transition:background .2s,color .2s;}}
      .mtc-btns button.active{{background:#2a9d8f;color:#fff;border-color:#2a9d8f;}}
      .mtc-svg{{width:100%;height:auto;display:block;}}
      .mtc-legend{{display:flex;flex-wrap:wrap;justify-content:center;gap:6px 14px;margin-top:2px;}}
      .mtc-leg{{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:#555;cursor:pointer;user-select:none;}}
      .mtc-leg-dot{{display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--c);}}
      .mtc-leg.off{{opacity:0.35;text-decoration:line-through;}}
      /* 统一按钮：点击/聚焦时去掉直角 outline，保持圆角 */
      button{{-webkit-tap-highlight-color:transparent;outline:none;}}
      button:focus,button:active{{outline:none;box-shadow:none;}}
      .mtc-btns button:active{{transform:scale(0.96);}}
    </style>
    <div class="mtc-wrap">
      <div class="mtc-title">各模块变化趋势（得分 / 正确率）</div>
      <div class="mtc-btns">
        <button id="mtc_s" class="active" onclick="mtcSwitch('s')">📊 模块得分</button>
        <button id="mtc_p" onclick="mtcSwitch('p')">📈 模块正确率</button>
      </div>
      <svg class="mtc-svg" viewBox="0 0 {W} {H}" onclick="mtcCloseAnnot()">
        {''.join(grid_html)}
        <line x1="{PL}" y1="{PT}" x2="{PL}" y2="{H-PB}" stroke="#ccc" stroke-width="1"/>
        <line x1="{PL}" y1="{H-PB}" x2="{W-PR}" y2="{H-PB}" stroke="#ccc" stroke-width="1"/>
        {''.join(paths_html)}
        {''.join(circles_html)}
        {''.join(xlabels_html)}
      </svg>
      <div class="mtc-legend">{''.join(legend_html)}</div>
    </div>
    <script>
      const MTC = {js_data};
      let mtcMode = 's';
      let mtcAnim = null;
      const mtcVisible = {{}};
      MTC.mods.forEach((m,i)=>mtcVisible[i]=true);

      function mtcPath(pts) {{
        let d = 'M ';
        for (let i=0;i<pts.length;i++) d += (i>0?' L ':'') + pts[i][0].toFixed(1) + ',' + pts[i][1].toFixed(1);
        return d;
      }}
      function mtcSwitch(mode) {{
        if (mode === mtcMode) return;
        mtcMode = mode;
        document.getElementById('mtc_s').classList.toggle('active', mode==='s');
        document.getElementById('mtc_p').classList.toggle('active', mode==='p');
        const from = mode==='s' ? MTC.pct_pts : MTC.score_pts;
        const to   = mode==='s' ? MTC.score_pts : MTC.pct_pts;
        const fromT = mode==='s' ? MTC.pct_ticks : MTC.score_ticks;
        const toT   = mode==='s' ? MTC.score_ticks : MTC.pct_ticks;
        const start = performance.now(), DUR = 700;
        if (mtcAnim) cancelAnimationFrame(mtcAnim);
        function step(now) {{
          const t = Math.min((now-start)/DUR, 1);
          const e = t<0.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2;
          for (let mi=0; mi<MTC.mods.length; mi++) {{
            const m = MTC.mods[mi], fp = from[m], tp = to[m];
            const cur = fp.map((p,i)=>[p[0], p[1]+(tp[i][1]-p[1])*e]);
            document.getElementById('mtc_path_'+mi).setAttribute('d', mtcPath(cur));
            for (let i=0;i<fp.length;i++) {{
              const c = document.getElementById('mtc_c_'+mi+'_'+i);
              if (c) c.setAttribute('cy', cur[i][1].toFixed(1));
            }}
          }}
          for (let i=0;i<fromT.length;i++) {{
            const v = fromT[i] + (toT[i]-fromT[i])*e;
            const el = document.getElementById('mtc_tick_'+i);
            if (el) el.textContent = v.toFixed(mode==='s'?1:0);
          }}
          if (t<1) mtcAnim = requestAnimationFrame(step);
        }}
        mtcAnim = requestAnimationFrame(step);
      }}
      function mtcToggle(mi) {{
        mtcVisible[mi] = !mtcVisible[mi];
        const op = mtcVisible[mi] ? 1 : 0.1;
        document.getElementById('mtc_path_'+mi).style.opacity = op;
        for (let i=0;i<MTC.x_labels.length;i++) {{
          const c = document.getElementById('mtc_c_'+mi+'_'+i);
          if (c) c.style.opacity = op;
        }}
        document.getElementById('mtc_leg_'+mi).classList.toggle('off', !mtcVisible[mi]);
      }}
      function mtcShowDetail(mi, idx) {{
        const ex = MTC.exams[idx];
        if (!ex) return;
        const old = document.getElementById('mtc_annot');
        if (old) old.remove();
        const anchor = document.getElementById('mtc_c_' + mi + '_' + idx);
        if (!anchor) return;
        const cx = parseFloat(anchor.getAttribute('cx'));
        const cy = parseFloat(anchor.getAttribute('cy'));
        const svg = document.querySelector('.mtc-svg');
        const ns = 'http://www.w3.org/2000/svg';
        // 描边/文字颜色 = 被点击模块的折线颜色
        const lineColor = document.getElementById('mtc_path_'+mi)?.getAttribute('stroke') || '#2a9d8f';
        // 找同一次考试中 y 坐标接近的所有模块点（交汇点，6px 内视为重合）
        const hitMods = [];
        for (let j=0;j<MTC.mods.length;j++) {{
          const c = document.getElementById('mtc_c_'+j+'_'+idx);
          if (!c) continue;
          const y = parseFloat(c.getAttribute('cy'));
          if (Math.abs(y - cy) < 6) hitMods.push(MTC.mods[j]);
        }}
        const lines = [ex.label];
        hitMods.forEach(m => {{
          lines.push(m + ': ' + ex.module_scores[m].toFixed(1) + '分 / ' + ex.module_accuracy[m].toFixed(0) + '%');
        }});
        const lineH = 16, padY = 7, padX = 8;
        // 用 getComputedTextLength 实际测量文字宽度，确保左右留白对称
        let maxTextW = 0;
        const probe = [];
        lines.forEach((line, i) => {{
          const t = document.createElementNS(ns, 'text');
          t.setAttribute('font-size', i === 0 ? '10.5' : '12');
          if (i >= 1) t.setAttribute('font-weight', '600');
          t.setAttribute('opacity', '0');
          t.textContent = line;
          svg.appendChild(t);
          maxTextW = Math.max(maxTextW, t.getComputedTextLength());
          probe.push(t);
        }});
        probe.forEach(t => t.remove());
        const labelW = Math.max(maxTextW + padX * 2, 100);
        const labelH = lines.length * lineH + padY * 2;
        const SVG_W = 720;
        let lx = cx + 12;
        if (lx + labelW > SVG_W - 5) lx = cx - labelW - 12;
        let ly = cy - labelH - 10;
        if (ly < 5) ly = cy + 14;
        const g = document.createElementNS(ns, 'g');
        g.setAttribute('id', 'mtc_annot');
        g.setAttribute('onclick', 'event.stopPropagation()');
        const rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', lx); rect.setAttribute('y', ly);
        rect.setAttribute('width', labelW); rect.setAttribute('height', labelH);
        rect.setAttribute('rx', 8); rect.setAttribute('ry', 8);
        rect.setAttribute('fill', 'rgba(255,255,255,0.97)');
        rect.setAttribute('stroke', lineColor); rect.setAttribute('stroke-width', '1.5');
        g.appendChild(rect);
        lines.forEach((line, i) => {{
          const t = document.createElementNS(ns, 'text');
          t.setAttribute('x', lx + padX);
          t.setAttribute('y', ly + padY + (i+1) * lineH - 3);
          t.setAttribute('font-size', i === 0 ? '10.5' : '12');
          t.setAttribute('fill', i === 0 ? '#666' : '#222');
          if (i >= 1) t.setAttribute('font-weight', '600');
          t.textContent = line;
          g.appendChild(t);
        }});
        svg.appendChild(g);
      }}
      function mtcCloseAnnot() {{
        const old = document.getElementById('mtc_annot');
        if (old) old.remove();
      }}
    </script>
    """
    return html


def _single_radar_animated(records) -> str:
    """自绘 SVG 个人单次考试雷达图：pills 按钮切换记录（不可手动输入），
    切换时雷达多边形顶点做 requestAnimationFrame 平滑过渡。"""
    if not records:
        return ""
    # 统一模块顺序
    all_mods = set()
    for r in records:
        all_mods.update((r.get("module_accuracy") or {}).keys())
    order = module_display_order(records)
    names = [m for m in order if m in all_mods] + [m for m in all_mods if m not in order]
    if not names:
        return ""

    import math
    W, H = 460, 360
    cx, cy, R = 230, 180, 108
    n = len(names)

    def _pts(acc):
        pts = []
        for i, m in enumerate(names):
            angle = -90 + 360 * i / n
            pct = float(acc.get(m, 0)) * 100
            r = R * pct / 100
            x = cx + r * math.cos(math.radians(angle))
            y = cy + r * math.sin(math.radians(angle))
            pts.append((round(x, 1), round(y, 1)))
        return pts

    # 100% 网格顶点 + 标签位置
    grid_pts, label_pts = [], []
    for i in range(n):
        angle = -90 + 360 * i / n
        gx = cx + R * math.cos(math.radians(angle))
        gy = cy + R * math.sin(math.radians(angle))
        grid_pts.append((round(gx, 1), round(gy, 1)))
        lx = cx + R * 1.2 * math.cos(math.radians(angle))
        ly = cy + R * 1.2 * math.sin(math.radians(angle))
        label_pts.append((round(lx, 1), round(ly, 1)))

    record_data = []
    for r in records:
        acc = r.get("module_accuracy") or {}
        scs = r.get("module_scores") or {}
        record_data.append({
            "id": r["id"],
            "label": f"#{r['id']} {r['name'] or ''} {(r['created_at'] or '')[:16]}",
            "pts": _pts(acc),
            "pcts": [round(float(acc.get(m, 0)) * 100, 2) for m in names],
            "scores": [round(float(scs.get(m, 0)), 2) for m in names],
        })

    # 背景网格（5 层）
    grid_html = []
    for layer in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = [(cx + (gx - cx) * layer, cy + (gy - cy) * layer) for gx, gy in grid_pts]
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"
        grid_html.append(f'<path d="{d}" fill="none" stroke="#e0e0e0" stroke-width="1"/>')
    axis_html = [f'<line x1="{cx}" y1="{cy}" x2="{gx}" y2="{gy}" stroke="#e0e0e0" stroke-width="1"/>'
                  for gx, gy in grid_pts]
    label_html = [f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="middle" '
                  f'font-size="11" fill="#555">{m}</text>'
                  for (x, y), m in zip(label_pts, names)]

    select_html = []
    for i, rd in enumerate(record_data):
        sel = "selected" if i == 0 else ""
        select_html.append(f'<option value="{i}" {sel}>{rd["label"]}</option>')

    init = record_data[0]
    data_path = "M " + " L ".join(f"{x},{y}" for x, y in init["pts"]) + " Z"
    circles_html = []
    for i, (x, y) in enumerate(init["pts"]):
        circles_html.append(
            f'<circle id="rad_c_{i}" cx="{x}" cy="{y}" r="4" fill="#2a9d8f" '
            f'stroke="#fff" stroke-width="1.5"><title>{names[i]}: {init["pcts"][i]:.2f}%</title></circle>')

    # 初始表格（第一条记录）
    tbody_html = []
    for i, m in enumerate(names):
        tbody_html.append(
            f'<tr><td>{m}</td><td>{init["pcts"][i]:.2f}%</td>'
            f'<td>{init["scores"][i]:.2f}</td></tr>')

    import json
    js_data = json.dumps({"names": names, "records": record_data}, ensure_ascii=False)

    html = f"""
    <style>
      .rad-wrap{{font-family:'Source Sans',sans-serif;background:#fff;border:1px solid #ececec;border-radius:8px;padding:10px;}}
      .rad-title{{font-size:14px;font-weight:600;color:#333;text-align:center;margin-bottom:6px;}}
      .rad-select{{width:100%;padding:5px 8px;border:1px solid #d0d0d0;border-radius:6px;font-size:12px;
                   background:#fff;color:#333;margin-bottom:8px;cursor:pointer;}}
      .rad-two-col{{display:flex;gap:14px;align-items:flex-start;}}
      .rad-left{{flex:1.3;min-width:0;}}
      .rad-right{{flex:1;min-width:0;}}
      .rad-svg{{width:100%;height:auto;display:block;}}
      .rad-table{{width:100%;border-collapse:collapse;font-size:12px;}}
      .rad-table th{{background:#f5f7f9;padding:6px 8px;text-align:left;border-bottom:2px solid #ddd;
                     font-weight:600;color:#444;white-space:nowrap;}}
      .rad-table td{{padding:5px 8px;border-bottom:1px solid #eee;color:#333;}}
      .rad-table tr:hover{{background:#f8f9fa;}}
      .rad-table .td-pct{{color:#2a9d8f;font-weight:500;}}
      .rad-table .td-score{{color:#e76f51;font-weight:500;}}
      @media(max-width:560px){{.rad-two-col{{flex-direction:column;}}}}
    </style>
    <div class="rad-wrap">
      <div class="rad-title">个人单次考试 · 模块正确率雷达图</div>
      <select class="rad-select" onchange="radSelect(parseInt(this.value))">
        {''.join(select_html)}
      </select>
      <div class="rad-two-col">
        <div class="rad-left">
          <svg class="rad-svg" viewBox="0 0 {W} {H}">
        {''.join(grid_html)}
        {''.join(axis_html)}
        {''.join(label_html)}
        <path id="rad_data" d="{data_path}" fill="rgba(42,157,143,0.2)" stroke="#2a9d8f" stroke-width="2"/>
        {''.join(circles_html)}
      </svg>
        </div>
        <div class="rad-right">
          <table class="rad-table">
            <thead><tr><th>模块</th><th>正确率</th><th>得分</th></tr></thead>
            <tbody id="rad_tbody">{''.join(tbody_html)}</tbody>
          </table>
        </div>
      </div>
    </div>
    <script>
      const RAD = {js_data};
      let radIdx = 0, radAnim = null;
      function radPath(pts){{return 'M '+pts.map(p=>p[0]+','+p[1]).join(' L ')+' Z';}}
      function radSelect(idx){{
        if(idx===radIdx) return;
        const from=RAD.records[radIdx].pts, to=RAD.records[idx].pts;
        const fp=RAD.records[radIdx].pcts, tp=RAD.records[idx].pcts;
        radIdx=idx;
        document.querySelector('.rad-select').value=idx;
        radUpdateTable(idx);
        const start=performance.now(), DUR=600;
        if(radAnim) cancelAnimationFrame(radAnim);
        (function step(now){{
          const t=Math.min((now-start)/DUR,1);
          const e=t<0.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
          const cur=from.map((p,i)=>[p[0]+(to[i][0]-p[0])*e, p[1]+(to[i][1]-p[1])*e]);
          document.getElementById('rad_data').setAttribute('d',radPath(cur));
          for(let i=0;i<cur.length;i++){{
            const c=document.getElementById('rad_c_'+i);
            if(c){{
              c.setAttribute('cx',cur[i][0].toFixed(1));
              c.setAttribute('cy',cur[i][1].toFixed(1));
              c.querySelector('title').textContent=RAD.names[i]+': '+(fp[i]+(tp[i]-fp[i])*e).toFixed(2)+'%';
            }}
          }}
          if(t<1) radAnim=requestAnimationFrame(step);
        }})(performance.now());
      }}
      function radUpdateTable(idx){{
        const rd=RAD.records[idx];
        let html='';
        for(let i=0;i<RAD.names.length;i++){{
          html+='<tr><td>'+RAD.names[i]+'</td>'+
                '<td class="td-pct">'+rd.pcts[i].toFixed(2)+'%</td>'+
                '<td class="td-score">'+rd.scores[i].toFixed(2)+'</td></tr>';
        }}
        document.getElementById('rad_tbody').innerHTML=html;
      }}
    </script>
    """
    return html


def module_display_order(records) -> list:
    """统一的模块显示顺序：按配置管理填写顺序（未配置的模块追加末尾）。

    两张雷达图（个人综合 / 个人单次）共用，保证各轴排列完全一致。
    """
    known = []
    for r in records:
        for m in (r.get("module_accuracy") or {}):
            if m not in known:
                known.append(m)
    return order_modules(known)


def overall_module_charts(records):
    """个人综合能力：基于某个人全部历史考试，聚合各模块平均正确率与平均得分。

    雷达图/柱状图按统一模块顺序（module_display_order）排列；
    柱状图内置「平均得分 / 平均正确率」切换按钮，用 plotly 帧动画实现
    柱子逐个涨跌的过渡效果。
    返回 (radar_fig, bar_fig, agg_df)；无数据返回 None。
    """
    rows = []
    for r in records:
        ma = r.get("module_accuracy") or {}
        ms = r.get("module_scores") or {}
        for m, acc in ma.items():
            try:
                rows.append({"模块": m, "正确率": float(acc), "得分": float(ms.get(m, 0))})
            except (TypeError, ValueError):
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    agg = df.groupby("模块").agg(
        考试次数=("正确率", "count"),
        平均正确率=("正确率", "mean"),
        平均得分=("得分", "mean"),
    ).reset_index()

    # 统一模块顺序：配置顺序优先，缺失模块追加到末尾
    order = module_display_order(records)
    known = set(agg["模块"])
    names = [m for m in order if m in known] + [m for m in agg["模块"] if m not in order]
    agg_idx = agg.set_index("模块").reindex(names).reset_index()

    vals = (agg_idx["平均正确率"] * 100).tolist()
    radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=names + [names[0]],
        fill="toself", line_color="#e76f51",
        hovertemplate="%{theta}: %{r:.2f}%<extra></extra>",
    ))
    radar.update_layout(
        title="个人综合能力：各模块平均正确率",
        polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=10))),
        height=400, showlegend=False)

    # 柱状图：自绘 div + CSS transition，切换「平均得分/平均正确率」时每根柱子平滑涨跌
    colors = px.colors.qualitative.Set2
    scores = agg_idx["平均得分"].tolist()
    pcts = (agg_idx["平均正确率"] * 100).tolist()
    s_max = max(scores) if scores else 10
    s_pct_h = [v / s_max * 100 for v in scores]
    bar_items = []
    for i, n in enumerate(names):
        c = colors[i % len(colors)]
        bar_items.append(f"""
        <div class="bar" style="--c:{c}">
          <div class="bar-val" id="val{i}">{scores[i]:.2f}</div>
          <div class="bar-fill" id="fill{i}" style="height:{s_pct_h[i]:.2f}%"></div>
          <div class="bar-label">{n}</div>
        </div>""")
    bar_html = f"""
    <style>
      *{{box-sizing:border-box;font-family:'Source Sans',sans-serif;}}
      .bc{{border:1px solid #ececec;border-radius:8px;padding:10px 6px 14px;background:#fff;}}
      .btns{{text-align:center;margin-bottom:8px;}}
      .btns button{{border:1px solid #d0d0d0;background:#fff;color:#444;padding:4px 14px;margin:0 3px;
                    border-radius:16px;cursor:pointer;font-size:13px;transition:background .2s,color .2s;}}
      .btns button.active{{background:#2a9d8f;color:#fff;border-color:#2a9d8f;}}
      .bars{{display:flex;align-items:flex-end;justify-content:space-around;
             height:250px;padding:0 4px;border-bottom:1px solid #ccc;position:relative;}}
      .bar{{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;
            height:100%;flex:1 1 0;min-width:0;position:relative;}}
      .bar-fill{{width:66%;max-width:66px;background:var(--c);border-radius:3px 3px 0 0;
                 min-height:1.5px;}}
      .bar-val{{font-size:12px;color:#666;margin-bottom:2px;white-space:nowrap;}}
      .bar-label{{font-size:11px;color:#333;margin-top:6px;text-align:center;white-space:nowrap;
                  overflow:hidden;text-overflow:ellipsis;max-width:100%;padding:0 2px;}}
    </style>
    <div class="bc">
      <div class="btns">
        <button id="b_s" class="active" onclick="animateTo('s')">📊 平均得分</button>
        <button id="b_p" onclick="animateTo('p')">📈 平均正确率</button>
      </div>
      <div class="bars">
        {''.join(bar_items)}
      </div>
    </div>
    <script>
      const S={json.dumps([round(v, 4) for v in scores])};
      const P={json.dumps([round(v, 4) for v in pcts])};
      const SMAX={s_max};
      const DUR=600;
      const easeInOutCubic=t=>t<0.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2;
      let _anim=null;
      function animateTo(m){{
        _anim && clearInterval(_anim);
        const arr = m==='s' ? S : P;
        const maxv = m==='s' ? SMAX : 100;
        const n=arr.length;
        const targets=arr.map(v=>Math.max(v/maxv*100, 0.8));
        const froms=[];
        for(let i=0;i<n;i++){{ const el=document.getElementById('fill'+i); froms.push(parseFloat(el.style.height)||0); }}
        document.getElementById('b_s').classList.toggle('active', m==='s');
        document.getElementById('b_p').classList.toggle('active', m==='p');
        const start=performance.now();
        _anim=setInterval(()=>{{
          const t=Math.min((performance.now()-start)/DUR, 1);
          const e=easeInOutCubic(t);
          for(let i=0;i<n;i++){{
            const el=document.getElementById('fill'+i);
            el.style.height=(froms[i]+(targets[i]-froms[i])*e)+'%';
            document.getElementById('val'+i).textContent =
              m==='s' ? arr[i].toFixed(2) : arr[i].toFixed(2)+'%';
          }}
          if(t>=1) clearInterval(_anim);
        }},16);
      }}
    </script>
    """
    # agg 返回按平均正确率降序（供强弱项表使用）
    agg = agg.sort_values("平均正确率", ascending=False).reset_index(drop=True)
    return radar, bar_html, agg


def init_state():
    st.session_state.setdefault("current", None)          # 识别结果
    st.session_state.setdefault("current_grad", None)     # 判卷结果
    st.session_state.setdefault("edited_df", None)        # 手动修正表
    st.session_state.setdefault("batch_df", None)
    # 当前卷标准答案：由用户上传的答案文件解析得到（每张答题卡答案不同）
    st.session_state.setdefault("session_answers", None)       # {题号: 选项}
    st.session_state.setdefault("session_answers_src", "")     # 来源文件名
    st.session_state.setdefault("session_answers_info", "")    # 解析信息
    st.session_state.setdefault("session_answers_digest", "")  # 内容指纹(避免重复解析)
    if "standard" not in st.session_state or "modules" not in st.session_state:
        std, mods = load_configs()
        st.session_state["standard"] = std
        st.session_state["modules"] = mods


def build_recognizer():
    return AnswerSheetRecognizer(
        fill_threshold=st.session_state.get("fill_threshold", 0.35),
        column_order=st.session_state.get("column_order", "ltr"),
        start_direction=st.session_state.get("start_direction", 1),
        registration_len=st.session_state.get("registration_len", 11),
        min_rows_per_band=3,
    )


# --------------------------------------------------------------------- #
# 当前卷标准答案：上传文件解析 + 识别前提醒
# --------------------------------------------------------------------- #
def current_answers() -> dict:
    """判卷使用的标准答案：优先用户上传的当前卷答案。"""
    return st.session_state.get("session_answers") or st.session_state["standard"]


def total_cfg() -> int:
    """由模块配置推导总题量。"""
    return max((m["end"] for m in st.session_state["modules"].values()), default=130)


def paper_total_questions(cur_std: dict | None = None) -> int:
    """当前卷判卷题量（不同地区/试卷类型题量不同）：
    优先取用户手动设置的「本卷总题量」；否则按答案文件最大题号自动判定。
    """
    manual = st.session_state.get("paper_total", 0)
    if manual:
        return int(manual)
    if cur_std:
        try:
            return max(int(k) for k in cur_std)
        except (ValueError, TypeError):
            pass
    return total_cfg()


def load_answers_from_upload(uploaded) -> bool:
    """解析上传的答案文件并写入 session；失败时清空并返回 False。"""
    try:
        answers, src = parse_answers_file(uploaded.name, uploaded.getvalue())
    except Exception as e:
        st.session_state["session_answers"] = None
        st.session_state["session_answers_digest"] = ""
        st.error(f"答案文件解析失败：{e}")
        return False
    if not answers:
        st.session_state["session_answers"] = None
        st.session_state["session_answers_digest"] = ""
        st.error("未能从文件中解析出答案（未识别到“题号+ABCD”格式），请检查文件内容。")
        return False
    st.session_state["session_answers"] = answers
    st.session_state["session_answers_src"] = uploaded.name
    st.session_state["session_answers_digest"] = hashlib.md5(uploaded.getvalue()).hexdigest()
    st.session_state["session_answers_info"] = (
        f"{src} · {len(answers)} 题（{min(answers)}-{max(answers)}）")
    st.success(f"✅ 已加载本卷标准答案：{st.session_state['session_answers_info']}")
    return True


def _upload_digest(up) -> str:
    return hashlib.md5(up.getvalue()).hexdigest()


def preview_answer_sources(ups):
    """对每个上传的答案来源（文件/拍照图片）用**多种识别方式**解析，
    每个非空识别方式作为一个候选。

    例如同一份 PDF → ["文字层提取", "OCR识别"]；同一张图片 → ["直接OCR", "增强OCR"]。
    返回 (previews, conflicts)。previews: [{name: "文件名〔方式〕", fmt, answers, error, file}]。
    """
    previews = []
    for u in ups:
        try:
            paths = parse_answers_file_multi(u.name, u.getvalue())
        except Exception as e:
            previews.append({"name": u.name, "fmt": "", "answers": {}, "error": str(e),
                             "file": u.name})
            continue
        if not paths:
            previews.append({"name": u.name, "fmt": "", "answers": {},
                             "error": "未能解析出答案（未识别到“题号+ABCD”格式）",
                             "file": u.name})
            continue
        for p in paths:
            previews.append({
                "name": f"{u.name}〔{p['name']}〕",
                "fmt": p["name"], "answers": p["answers"], "error": None,
                "file": u.name,
            })
    return previews, detect_answer_conflicts(previews)


def detect_answer_conflicts(previews) -> list:
    """多来源重叠题号但选项不同 → 冲突列表 [{题号, 各来源: {名: 选项}}]。"""
    by_q: dict = {}
    for p in previews:
        if not p["answers"]:
            continue
        for q, v in p["answers"].items():
            by_q.setdefault(q, {})[p["name"]] = v
    conflicts = []
    for q, srcs in by_q.items():
        if len(set(srcs.values())) > 1:
            conflicts.append({"题号": q, "各来源": srcs})
    return sorted(conflicts, key=lambda c: c["题号"])


def merge_answer_sources(previews, chosen_name: str | None = None) -> dict:
    """合并多来源答案：无冲突题直接合并；冲突题以 chosen_name 来源为准。"""
    merged: dict = {}
    for p in previews:
        if not p["answers"]:
            continue
        for q, v in p["answers"].items():
            if q not in merged or (chosen_name and p["name"] == chosen_name):
                merged[q] = v
    return dict(sorted(merged.items()))


def _source_desc(previews, ok_names) -> str:
    return "+".join(p["name"] for p in previews if p["name"] in ok_names)


def answer_uploader_block(container, key, label):
    """渲染答案上传控件。

    对同一份答案文件用**多种识别方式**解析做交叉验证：
      - PDF → 文字层提取 / OCR识别
      - 图片 → 直接OCR / 增强OCR
      - Excel → 结构化表格 / 文本模式
    各方式结果一致则可信直接采用；不一致（冲突）时展示冲突表，
    让用户选择以哪种识别方式为准。

    延迟解析：上传后先“只读预览”，识别按钮点击时才正式采用（resolve_answers）。
    这样可区分“本次新上传的答案”与控件里保留的旧答案文件——只有前者才视为
    已上传新答案，防止用上一卷的答案判本卷。
    """
    with container:
        ups = st.file_uploader(
            label,
            type=["xlsx", "xls", "docx", "pdf", "txt", "csv",
                  "jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key=key,
            help="每张答题卡答案不同，请上传本次试卷标准答案。支持 Excel/Word/PDF/TXT/CSV，"
                 "也支持拍照/扫描的答案图片（自动 OCR）。同一份答案会用多种识别方式交叉验证"
                 "（PDF=文字层/OCR，图片=直接/增强OCR），结果不一致时提示你选择以哪种为准。")
        # 防御：下方预览/冲突解析全部包 try/except，任何异常都只提示而不拖垮页面
        # （否则上传区抛异常会让「识别并批改」按钮无法渲染，表现为“按钮消失”）
        try:
            cur_digest = st.session_state.get("session_answers_digest", "")
            previews = None
            if ups:
                new_ups = [u for u in ups if _upload_digest(u) != cur_digest]
                if new_ups:
                    previews, conflicts = preview_answer_sources(ups)
                    st.session_state[f"{key}_previews"] = previews
                    st.session_state[f"{key}_conflicts"] = conflicts
                previews = st.session_state.get(f"{key}_previews")
                conflicts = st.session_state.get(f"{key}_conflicts", [])
                # 各识别方式解析结果
                for p in (previews or []):
                    if p["error"]:
                        st.error(f"❌ {p['name']}：{p['error']}")
                    elif p["answers"]:
                        st.info(f"📄 {p['name']} → {len(p['answers'])} 题"
                                f"（{min(p['answers'])}-{max(p['answers'])}）")
                    else:
                        st.error(f"❌ {p['name']}：未能解析出答案")
                # 答案有效范围：排除答案中多余的题（如差异题/超出本卷题量）
                if previews and any(p["answers"] for p in previews):
                    rng = st.text_input(
                        "答案有效范围（默认全部；多余题排除。示例：`1-130` 或 `1-110,120-130`；"
                        "排除指定区间用 `排除:131-140`）",
                        key=f"{key}_range", placeholder="留空 = 采用答案全部题目",
                        help="答案文件里可能含本卷之外的题（如差异题页、整套 1-140 但本卷 130 题）。"
                             "设置有效范围后，范围外的题号将被忽略，不计入判卷。")
                # 多种识别方式结果不一致：让用户选择基准 + 逐题手动修改
                if conflicts:
                    ok_sources = [p["name"] for p in (previews or []) if p["answers"]]
                    st.warning(f"⚠️ 同一答案的不同识别方式有 **{len(conflicts)}** 处不一致"
                               f"（如题 {conflicts[0]['题号']}…），请选择基准后逐题确认：")
                    chosen = st.radio(
                        "以哪种识别结果为准（冲突题可在下方表格逐题修改）", ok_sources,
                        index=0, key=f"{key}_chosen",
                        help="无冲突的题号仍会自动合并所有方式的结果。")
                    rows = []
                    for c in conflicts:
                        row = {"题号": c["题号"]}
                        for src_name, val in c["各来源"].items():
                            row[src_name] = val
                        row["最终采用"] = c["各来源"].get(chosen, "")
                        rows.append(row)
                    cf = pd.DataFrame(rows)
                    col_config = {
                        "题号": st.column_config.NumberColumn("题号", disabled=True),
                        "最终采用": st.column_config.SelectboxColumn(
                            "最终采用(可修改)", options=["", "A", "B", "C", "D"],
                            help="手动指定该题最终答案；留空则该题按「无此答案」处理。"),
                    }
                    for src_name in ok_sources:
                        col_config[src_name] = st.column_config.TextColumn(src_name, disabled=True)
                    st.caption("👇 冲突题逐题确认：在「最终采用(可修改)」列下拉选择/修改该题答案")
                    st.data_editor(cf, key=f"{key}_conflict_editor", width="stretch",
                                   hide_index=True, column_config=col_config)
            elif st.session_state.get("session_answers"):
                st.caption("📌 当前已加载答案："
                           f"{st.session_state.get('session_answers_info')}。")
            else:
                st.caption("尚未加载答案，识别前必须先上传本卷标准答案。")
        except Exception as e:
            st.warning(f"⚠️ 答案解析出现异常：{e}（请检查答案文件格式后重试）")
    return ups


def conflict_override(key: str) -> dict:
    """读取用户在冲突表里逐题手动指定的「最终采用」→ {题号: 选项}。"""
    raw = st.session_state.get(f"{key}_conflict_editor")
    if raw is None:
        return {}
    try:
        # st.data_editor 的 session_state 是 DataEditorState（用 .data 取 df）
        df = raw.data if hasattr(raw, "data") else raw
    except Exception:
        return {}
    if df is None or (hasattr(df, "empty") and df.empty):
        return {}
    override: dict = {}
    try:
        for _, r in df.iterrows():
            try:
                q = int(r["题号"])
            except (ValueError, TypeError):
                continue
            v = str(r.get("最终采用", "")).strip().upper()
            if v in OPTIONS:
                override[q] = v
    except Exception:
        return {}
    return override


def filter_answers_by_range(answers: dict, range_str: str) -> dict:
    """按用户设置的有效范围过滤答案，排除多余题（如差异题、超出本卷范围的题号）。

    range_str 格式：
      - 空 / "全部"：不过滤
      - "1-130"：只保留题号 1~130
      - "1-110,120-130"：多段区间（逗号/空格/分号分隔）
      - "排除:131-140" / "排除131-140"：排除指定区间，其余保留
    """
    s = (range_str or "").strip()
    if not s or s in ("全部", "all", "All", "ALL"):
        return answers
    exclude = False
    low = s.lower()
    if low.startswith("排除") or low.startswith("exclude"):
        exclude = True
        s = low
        for w in ("排除", "exclude", ":", "："):
            s = s.replace(w, " ")
    allowed = set()
    for part in re.split(r"[\s,，;；、]+", s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d{1,3})\s*[-~～至]{1,2}\s*(\d{1,3})$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            allowed.update(range(min(a, b), max(a, b) + 1))
        elif re.fullmatch(r"\d{1,3}", part):
            allowed.add(int(part))
    if not allowed:
        return answers
    if exclude:
        return {q: c for q, c in answers.items() if q not in allowed}
    return {q: c for q, c in answers.items() if q in allowed}


def resolve_answers(ups, chosen_name: str | None = None, override: dict | None = None) -> dict | None:
    """识别时确定本卷标准答案：控件内有新文件则解析并更新；否则沿用已加载。

    多识别方式时：成功解析的识别结果合并（冲突题以 chosen_name 为准），
    再用 override（用户在冲突表逐题手改的最终值）覆盖。
    返回 None 表示无法获得有效答案（应阻断识别）。
    """
    if ups:
        cur_digest = st.session_state.get("session_answers_digest", "")
        new_ups = [u for u in ups if _upload_digest(u) != cur_digest]
        if new_ups:
            previews, _conflicts = preview_answer_sources(ups)
            ok = [p for p in previews if p["answers"]]
            if not ok:
                st.error("未能从上传的文件中解析出答案（未识别到“题号+ABCD”格式）。")
                return None
            answers = merge_answer_sources(previews, chosen_name)
            if override:
                answers.update(override)
            if not answers:
                st.error("未能从上传的文件中解析出答案。")
                return None
            src_desc = _source_desc(previews, {p["name"] for p in ok})
            st.session_state["session_answers"] = answers
            st.session_state["session_answers_src"] = src_desc
            st.session_state["session_answers_digest"] = "-".join(
                sorted(_upload_digest(u) for u in ups))
            st.session_state["session_answers_info"] = (
                f"{src_desc} · {len(answers)} 题（{min(answers)}-{max(answers)}）")
            # 每次上传新答案 → 自动覆盖该用户配置中的标准答案（最新答案为准，
            # 防止刷新/换设备后回落旧答案）；写配置失败不阻断判分
            try:
                user = current_user()
                if user:
                    save_standard_answers(
                        {str(k): v for k, v in answers.items()}, user=user)
                    st.session_state["standard"] = {
                        str(k): v for k, v in answers.items()}
            except Exception:
                pass
            return answers
    return st.session_state.get("session_answers")


def answers_ready(ups, reuse: bool) -> bool:
    """识别前检查：控件里有新答案文件（识别时解析）或显式勾选沿用。"""
    if ups:
        cur_digest = st.session_state.get("session_answers_digest", "")
        if any(_upload_digest(u) != cur_digest for u in ups):
            return True
    return bool(reuse and st.session_state.get("session_answers"))


def answers_coverage_warning(answers: dict, total_cfg: int):
    """答案题数不足时提示。"""
    n = len(answers)
    if n and n < total_cfg:
        missing = total_cfg - n
        st.warning(f"⚠️ 本卷答案只有 **{n}** 题，配置题量为 **{total_cfg}** 题，"
                   f"缺 **{missing}** 题（按错误计）。请确认是否上传了完整答案。")


# --------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------- #
def render_login() -> str:
    """用户登录 / 工作区切换，返回当前登录用户名（未登录返回空）。

    每个用户完全隔离：配置管理、标准答案、历史记录都按用户独立。
    用户名按浏览器 localStorage 各自记忆（每台电脑各记各的）。
    """
    with st.sidebar:
        st.header("🔐 用户工作区")
        if st.session_state.get("logged_in"):
            u = str(st.session_state.get("current_user", "")).strip()
            st.success(f"当前用户：**{u}**")
            if st.button("退出登录"):
                st.session_state["logged_in"] = False
                st.session_state["current_user"] = ""
                # 清除 URL 用户参数与本地记忆，避免刷新后自动重登
                st.query_params.pop("user", None)
                components.html("<script>localStorage.removeItem('xingce_user');</script>", height=0)
                std, mods = load_configs("")
                st.session_state["standard"] = std
                st.session_state["modules"] = mods
                st.rerun()
            return u
        # 未登录：用户名从浏览器 localStorage 带出。
        # st.html / st.markdown 直接注入的 <script> 在 React 下不会执行；
        # components.html 的 iframe 沙箱缺 allow-top-navigation 无法改写父窗口 URL。
        # 方案：手动注入 srcdoc iframe（带 allow-same-origin+allow-scripts+allow-top-navigation），
        # 由浏览器原生执行 iframe 内脚本，从共享 localStorage 读用户名并导航父窗口到 ?user=xxx。
        _raw = st.query_params.get("user", "")
        url_user = _raw[0] if isinstance(_raw, list) else (_raw or "")
        st.markdown("""
        <iframe sandbox="allow-same-origin allow-scripts allow-top-navigation"
          srcdoc="<script>(function(){try{var ls=localStorage.getItem('xingce_user')||'';var p=new URLSearchParams(window.parent.location.search);if(p.has('user')){localStorage.setItem('xingce_user',p.get('user')||'');}else if(ls){var b=window.parent.location.href.split('?')[0];window.parent.location.href=b+'?user='+encodeURIComponent(ls);}}catch(e){}})();</script>"
          style="display:none"></iframe>
        """, unsafe_allow_html=True)
        # 刷新自动恢复：本机已记忆且未设密码的用户直接进入（每台电脑各自记忆）。
        # 设置了密码的用户必须重新输入密码，不自动登录。
        # 刷新自动恢复：本机已记忆该用户 → 直接进入（每台电脑各自记忆，免重复登录）。
        # 已设置密码的用户：本机记忆过仍自动进入（本人设备便捷），新设备/未记忆则需输密码。
        if url_user and user_exists(url_user):
            st.session_state["logged_in"] = True
            st.session_state["current_user"] = url_user
            std, mods = load_configs(url_user)
            st.session_state["standard"] = std
            st.session_state["modules"] = mods
            st.rerun()
        login_user = st.text_input("用户名", value=url_user, key="login_user",
                                   placeholder="首次输入即创建该用户工作区")
        login_pwd = st.text_input("密码（未设置可留空）", type="password", key="login_pwd")
        if st.button("进入工作区"):
            u = login_user.strip()
            if not u:
                st.warning("请输入用户名")
            elif user_exists(u) and not user_check_password(u, login_pwd):
                st.error("密码错误")
            else:
                if not user_exists(u):
                    user_set_password(u, login_pwd)
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = u
                std, mods = load_configs(u)
                st.session_state["standard"] = std
                st.session_state["modules"] = mods
                st.query_params["user"] = u
                st.rerun()
        st.caption("每个用户独立：配置管理、标准答案、历史记录完全隔开。")
        return ""


def render_sidebar():
    with st.sidebar:
        cur_user = render_login()
        st.markdown("---")
        st.header("⚙️ 识别参数")
        st.session_state["fill_threshold"] = st.slider(
            "填涂判定阈值（涂黑占比）", 0.20, 0.60, 0.35, 0.01,
            help="行测涂卡要求“深满匀”，超过该占比判为选中。")
        st.session_state["column_order"] = st.selectbox(
            "列序方向", ["ltr", "rtl"], format_func=lambda x: "左→右" if x == "ltr" else "右→左")
        st.session_state["start_direction"] = st.radio(
            "最左列读数方向", [1, -1],
            format_func=lambda x: "自上而下" if x == 1 else "自下而上",
            help="纵向S型：奇数列自上而下、偶数列自下而上。若整体映射反向请切换。")
        st.session_state["registration_len"] = st.number_input(
            "准考证号位数(0=不限制)", 0, 15, 11)
        st.session_state["paper_total"] = st.number_input(
            "本卷总题量(0=按答案自动)", 0, 300, 0, step=5,
            help="不同地区/试卷类型题量不同（130/135/140…）。默认按答案文件题数自动判定；"
                 "也可手动指定总题量，超出模块配置范围的题会按最后一模块权重自动计分。")
        st.checkbox(
            "总分折算百分制(满分100)", value=False, key="score_norm",
            help="不勾选（默认）：模块得分 = 答对数 × 单题权重，总分 = 各模块得分之和，"
                 "满分 = Σ(题数×权重)。勾选：把总分折算到 100 分（模块得分同步折算），"
                 "适合想让满分固定为 100 分的场景。")
        st.markdown("---")
        if st.button("重新加载配置(JSON)"):
            std, mods = load_configs(current_user())
            st.session_state["standard"] = std
            st.session_state["modules"] = mods
            st.toast("配置已重新加载")
        st.caption("配置位于 configs/ 目录（已登录时按用户独立），也可在「配置管理」页编辑。")


# --------------------------------------------------------------------- #
# Tab1 单张识别
# --------------------------------------------------------------------- #
def render_single():
    st.subheader("① 上传答题卡照片")
    upload = st.file_uploader("上传答题卡图片 (jpg/png)", type=["jpg", "jpeg", "png"],
                              key="single_upload")

    st.markdown("---")
    st.subheader("② 上传本卷标准答案")
    c_reuse, _ = st.columns([1.6, 2])
    reuse = c_reuse.checkbox("本卷与已加载答案是同一套（沿用，不重新上传）", value=False)
    ans_up = answer_uploader_block(
        st.container(), "ans_upload_single",
        "📄 标准答案文件（Excel / Word / PDF / TXT）")

    if not upload:
        st.info("请上传一张行测答题卡照片。支持纵向S型排列（题号竖排、蛇形走位）。")
        return

    # 关键提醒：每张答题卡答案不同，未上传新答案时醒目提示并阻断识别
    if not answers_ready(ans_up, reuse):
        st.error("⚠️ **尚未上传本次试卷的标准答案。** 由于每张答题卡的答案不同，"
                 "请先在上方上传本卷答案（Excel/Word/PDF/TXT）后再识别；"
                 "若确与当前已加载答案是同一套，请勾选“沿用”。",
                 icon="🚫")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🚀 识别并批改", type="primary", width="stretch"):
            if not answers_ready(ans_up, reuse):
                st.error("请先上传本卷标准答案，或勾选“沿用已加载答案”。")
                return
            img = load_image(upload)
            if img is None:
                st.error("图片读取失败，请更换图片。")
                return
            cur_std = resolve_answers(
                ans_up,
                st.session_state.get("ans_upload_single_chosen"),
                conflict_override("ans_upload_single"))
            if not cur_std:
                return
            cur_std = filter_answers_by_range(
                cur_std, st.session_state.get("ans_upload_single_range", ""))
            if not cur_std:
                st.error("按有效范围过滤后没有可用答案，请检查「答案有效范围」设置。")
                return
            paper_total = paper_total_questions(cur_std)
            answers_coverage_warning(cur_std, paper_total)
            with st.spinner("正在识别（透视矫正 → 气泡检测 → S型映射 → 判卷）..."):
                rec = build_recognizer()
                res = rec.recognize(img)
                del img
                gc.collect()
                grad = grade(res.answers, cur_std,
                             st.session_state["modules"],
                             normalize_to_100=st.session_state.get("score_norm", False),
                             total_questions=paper_total)
                st.session_state["current"] = res
                st.session_state["current_grad"] = grad
                st.session_state["edited_df"] = answers_df(res.answers, cur_std)
                st.session_state["current_img_name"] = upload.name
            st.success("识别完成")

    cur = st.session_state.get("current")
    grad = st.session_state.get("current_grad")
    if cur is None or grad is None:
        return

    # 题量与识别结果不一致提示（题量由答案文件或手动设置决定）
    total = paper_total_questions(st.session_state.get("session_answers"))
    if cur.detected_questions != total:
        st.warning(
            f"⚠️ 检测到 **{cur.detected_questions}** 题（每列{cur.rows_per_band}题×{cur.band_count}列），"
            f"与当前卷题量 **{total}**（由答案文件/手动设置）不符，题号映射可能错位。"
            "请检查答题卡版式；如为非常规排列，请在下方手动修正。")
    if cur.warning:
        st.warning("⚠️ " + cur.warning)

    with col_b:
        # 矫正后图像 + 判定叠加图
        st.markdown("**透视矫正后图像（绿=正确 / 红=错误 / 灰=未涂）**")
        if cur.warped is not None:
            std = current_answers()
            drawn = AnswerSheetRecognizer().draw_answers(cur, std)
            st.image(bgr2rgb(drawn), width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("总分", f"{grad.total_score:.2f} 分")
    m2.metric("答对", f"{grad.correct_count} / {grad.total_count}")
    m3.metric("准考证号", cur.registration or "未识别")
    st.caption(
        f"🔍 识别诊断：版式 {cur.rows_per_band} 行 × {cur.band_count} 列 = {cur.detected_questions} 题"
        f" ｜ 填涂点 {len(cur.bubbles)} 个"
        f" ｜ {cur.warning or '无异常'}")

    # 错题复盘模块
    st.markdown("### ❌ 错题复盘")
    if grad.wrong_details:
        q2mod = question_to_module(st.session_state["modules"])
        wrong_df = pd.DataFrame([{
            "题号": d["题号"],
            "模块": q2mod.get(d["题号"], ("未知",))[0],
            "识别选项": d["识别"],
            "正确答案": d["标准"],
        } for d in sorted(grad.wrong_details, key=lambda x: x["题号"])])
        # 错题统计按「配置管理」模块顺序展示（而非 value_counts 出现次数降序）
        # 注意：modules 是 {模块名: {start,end,weight,...}} 的 dict，模块名即 key
        mod_order = list(st.session_state["modules"].keys())
        mod_counts = wrong_df["模块"].value_counts().to_dict()
        mod_stat = " ｜ ".join(
            f"{m} 错 {mod_counts.get(m, 0)} 题" for m in mod_order
            if mod_counts.get(m, 0))
        st.caption(f"共错 **{len(wrong_df)}** 题：{mod_stat}")
        st.dataframe(wrong_df, width="stretch", hide_index=True)
    else:
        st.success("🎉 全部答对，无错题！")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.plotly_chart(module_bar(grad), width="stretch")
    with c2:
        st.plotly_chart(radar_chart(grad), width="stretch")

    # 手动修正模式
    st.markdown("### 手动修正（可点击修改“识别选项”后重新计分）")
    edited = st.data_editor(
        st.session_state.get("edited_df", pd.DataFrame()),
        num_rows="fixed",
        width="stretch",
        height=360,
        column_config={
            "题号": st.column_config.NumberColumn("题号", disabled=True),
            "识别选项": st.column_config.SelectboxColumn("识别选项", options=["", "A", "B", "C", "D"]),
            "标准答案": st.column_config.TextColumn("标准答案", disabled=True),
            "判定": st.column_config.TextColumn("判定", disabled=True),
        },
        key="editor",
    )
    if st.button("🔄 按修正结果重新计分"):
        answers = df_to_answers(edited)
        st.session_state["current"].answers = answers
        st.session_state["edited_df"] = edited
        st.session_state["current_grad"] = grade(
            answers, current_answers(), st.session_state["modules"],
            normalize_to_100=st.session_state.get("score_norm", False))
        st.success("已重新计分")

    # 保存记录
    st.markdown("### 💾 保存到历史库")
    with st.form("save_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        name = c1.text_input("考生姓名", value=current_user(),
                             help="已预填当前登录用户名，可修改。")
        reg = c2.text_input("准考证号（自动识别，可改）",
                            value=cur.registration or "")
        if st.form_submit_button("保存记录", type="primary"):
            if not current_user():
                st.error("请先登录后再保存记录（未登录不写入历史库，避免数据混入全局）。")
            else:
                db = RecordDB()
                rid = db.insert(
                    owner=current_user(),
                    name=name.strip(), registration=reg.strip(), paper_code="",
                    total_score=st.session_state["current_grad"].total_score,
                    module_scores=st.session_state["current_grad"].module_scores,
                    module_accuracy=st.session_state["current_grad"].module_accuracy,
                    wrong_details=st.session_state["current_grad"].wrong_details,
                    recognized_answers={str(k): v for k, v in
                                        st.session_state["current_grad"].detail.items()
                                        if v.get("rec")},
                )
                st.success(f"已保存记录 #{rid} → 历史统计页查看")


# --------------------------------------------------------------------- #
# Tab2 批量识别
# --------------------------------------------------------------------- #
def render_batch():
    st.subheader("批量导入多张答题卡（同一考生多科 / 多人，一键批量批改）")
    files = st.file_uploader("可多选图片", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="batch_upload")
    default_name = st.text_input("默认考生姓名（可为空，逐行可改）", value="")

    st.markdown("---")
    st.subheader("① 上传本卷标准答案")
    c_reuse, _ = st.columns([1.6, 2])
    reuse = c_reuse.checkbox("批量卡均为同一套已加载答案（沿用，不重新上传）", value=False)
    ans_up = answer_uploader_block(
        st.container(), "ans_upload_batch",
        "📄 标准答案文件（Excel / Word / PDF / TXT）——整批使用同一本卷答案")

    if not files:
        return

    if not answers_ready(ans_up, reuse):
        st.error("⚠️ **尚未上传本卷标准答案。** 每张答题卡的答案不同，"
                 "批量批改前请先上传本卷答案；若整批确为已加载的同一套答案，请勾选“沿用”。",
                 icon="🚫")

    if st.button("📚 批量识别并批改", type="primary"):
        if not answers_ready(ans_up, reuse):
            st.error("请先上传本卷标准答案，或勾选“沿用已加载答案”。")
            return
        cur_std = resolve_answers(
            ans_up,
            st.session_state.get("ans_upload_batch_chosen"),
            conflict_override("ans_upload_batch"))
        if not cur_std:
            return
        cur_std = filter_answers_by_range(
            cur_std, st.session_state.get("ans_upload_batch_range", ""))
        if not cur_std:
            st.error("按有效范围过滤后没有可用答案，请检查「答案有效范围」设置。")
            return
        paper_total = paper_total_questions(cur_std)
        answers_coverage_warning(cur_std, paper_total)
        # 服务器内存有限（1.6G），批量一次最多 5 张，防止内存爆掉拖垮服务
        if len(files) > 5:
            st.error("服务器资源有限，批量识别一次最多 5 张，请分批上传（已选 %d 张）。" % len(files))
            return
        rows, results = [], []
        progress = st.progress(0.0)
        rec = build_recognizer()
        for i, f in enumerate(files):
            img = load_image(f)
            if img is None:
                rows.append({"文件名": f.name, "状态": "读取失败"})
                continue
            res = rec.recognize(img)
            del img
            gc.collect()
            grad = grade(res.answers, cur_std,
                         st.session_state["modules"],
                         normalize_to_100=st.session_state.get("score_norm", False),
                         total_questions=paper_total)
            rows.append({
                "文件名": f.name,
                "准考证号": res.registration,
                "总分": round(grad.total_score, 2),
                "答对": f"{grad.correct_count}/{grad.total_count}",
                "版式": f"{res.rows_per_band}×{res.band_count}",
                "警告": res.warning[:40],
            })
            results.append((f.name, res, grad))
            progress.progress((i + 1) / len(files))
        df = pd.DataFrame(rows)
        st.session_state["batch_df"] = df
        st.session_state["batch_results"] = results
        st.success(f"批量识别完成，共 {len(rows)} 张")

    df = st.session_state.get("batch_df")
    if df is not None:
        st.dataframe(df, width="stretch", hide_index=True)
        # 逐张明细
        with st.expander("查看每张答题卡明细"):
            for fname, res, grad in st.session_state.get("batch_results", []):
                st.markdown(f"**{fname}**  | 总分 {grad.total_score:.2f} | 准考证号 {res.registration}")
                c1, c2 = st.columns(2)
                c1.plotly_chart(module_bar(grad), width="stretch", key=f"bm_{fname}")
                c2.plotly_chart(radar_chart(grad), width="stretch", key=f"br_{fname}")

        st.markdown("### 保存全部到历史库")
        save_base = df.drop(columns=["答对", "版式", "警告"], errors="ignore").copy()
        save_base.insert(0, "姓名", default_name)
        save_df = st.data_editor(
            save_base, num_rows="dynamic", width="stretch", key="batch_save_df")
        if st.button("💾 一键保存全部"):
            if not current_user():
                st.error("请先登录后再批量保存（未登录不写入历史库）。")
            else:
                db = RecordDB()
                batch_id = f"B{int(time.time())}"
                n = 0
                for i, row in save_df.iterrows():
                    name = str(row.get("姓名", "") or default_name)
                    fname = row["文件名"]
                    # 找到对应结果
                    for fn, res, grad in st.session_state.get("batch_results", []):
                        if fn == fname:
                            db.insert(
                                owner=current_user(),
                                name=name.strip(), registration=str(row.get("准考证号", "") or ""),
                                paper_code="",
                                total_score=grad.total_score,
                                module_scores=grad.module_scores,
                                module_accuracy=grad.module_accuracy,
                                wrong_details=grad.wrong_details,
                                recognized_answers={str(k): v for k, v in grad.detail.items() if v.get("rec")},
                                batch_id=batch_id,
                            )
                            n += 1
                            break
                st.success(f"已保存 {n} 条记录（批次 {batch_id}）")


# --------------------------------------------------------------------- #
# Tab3 历史统计
# --------------------------------------------------------------------- #
def render_history():
    st.subheader("历史记录与统计分析")
    cur_user = current_user()
    if not cur_user:
        st.warning("⚠️ 请先登录（左侧「用户工作区」输入用户名进入）后，才能查看历史记录。"
                   "每个用户的数据完全隔离，未登录不显示任何记录。")
        return
    db = RecordDB()

    s1, s2, s3 = st.columns([1.5, 1, 1])
    search = s1.text_input("搜索（姓名/准考证号）", value="")
    if s2.button("🔄 刷新"):
        pass

    # 读取记录并按「当前模块配置」实时重算分数（修改配置后历史分数自动变化）
    records = db.list_records(search=search, owner=cur_user)
    _modules = st.session_state.get("modules") or {}
    for r in records:
        res = recalc_from_record(r, _modules)
        if res is not None:
            r["total_score"] = res.total_score
            r["module_scores"] = res.module_scores
            r["module_accuracy"] = res.module_accuracy

    # 统计概览（用重算后的分数）
    if records:
        _scores = [r["total_score"] for r in records]
        _avg = sum(_scores) / len(_scores)
        _max = max(_scores)
    else:
        _avg = _max = 0
    s3.caption(f"累计 {len(records)} 条 | 平均 {_avg:.2f} 分 | 最高 {_max:.2f} 分"
               + f" ｜ 当前用户：{cur_user}")

    if not records:
        st.info("暂无历史记录。请先在「单张识别 / 批量识别」中保存。")
        return

    disp = pd.DataFrame([{
        "id": r["id"], "姓名": r["name"], "准考证号": r["registration"],
        "总分": r["total_score"], "时间": r["created_at"],
    } for r in records])
    st.dataframe(disp, width="stretch", hide_index=True)

    # 删除
    with st.expander("🗑 删除记录（按 id）"):
        del_id = st.number_input("要删除的记录 id", 1, int(1e9), 1)
        if st.button("删除该记录"):
            if db.delete(int(del_id), owner=cur_user):
                st.success(f"已删除 #{int(del_id)}")
                st.rerun()
            else:
                st.error("未找到该 id" + ("（或该记录不属于当前用户）" if cur_user else ""))

    st.markdown("---")
    # 考生选择（默认全部记录；选中后只看该考生）
    cands = sorted({r["registration"] or r["name"] for r in records} | {r["name"] for r in records if r["name"]})
    cand = st.selectbox("选择考生（用于趋势分析）", ["(全部记录)"] + cands)
    if cand and cand != "(全部记录)":
        reg_cand = cand if cand in {r["registration"] for r in records if r["registration"]} else ""
        if reg_cand:
            recs = [r for r in records if r["registration"] == reg_cand]
        else:
            recs = [r for r in records if r["name"] == cand]
        if not recs:
            recs = records
    else:
        recs = records

    if recs:
        # 总分变化趋势
        st.plotly_chart(trend_chart(recs), width="stretch")
        # 各模块变化趋势（得分 / 正确率 切换，自绘 SVG 平滑过渡动画）
        _mtc_html = module_trend_animated_chart(recs)
        if _mtc_html:
            components.html(_mtc_html, height=600)

    # 个人综合能力雷达图：基于该个人的所有考试推断（区别于上方按考生筛选的趋势）
    st.markdown("---")
    st.subheader("📊 个人综合能力（基于TA的所有考试）")
    persons = []
    seen = set()
    for r in records:
        p = r["registration"] or r["name"] or "未知"
        if p not in seen:
            seen.add(p)
            persons.append(p)
    if persons:
        default_p = records[0]["registration"] or records[0]["name"] or "未知"
        sel_p = st.selectbox(
            "选择个人（综合能力基于TA的全部历史考试推断）", persons,
            index=persons.index(default_p) if default_p in persons else 0,
            format_func=lambda p: p)
        recs_person = [r for r in records if (r["registration"] or r["name"] or "未知") == sel_p]
        overall = overall_module_charts(recs_person)
        if overall:
            radar_o, bar_o, agg = overall
            tot = [r["total_score"] for r in recs_person]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("考试次数", len(recs_person))
            c2.metric("平均总分", f"{sum(tot) / len(tot):.2f}")
            c3.metric("最高总分", f"{max(tot):.2f}")
            c4.metric("最低总分", f"{min(tot):.2f}")
            col1, col2 = st.columns(2)
            col1.plotly_chart(radar_o, width="stretch")
            with col2:
                st.components.v1.html(bar_o, height=440)
            agg_disp = pd.DataFrame({
                "模块": agg["模块"],
                "考试次数": agg["考试次数"],
                "平均正确率(%)": agg["平均正确率"] * 100,
                "平均得分(分)": agg["平均得分"],
            })
            st.caption(f"{sel_p} 按平均正确率降序排列 = 强 → 弱项")
            st.dataframe(agg_disp, width="stretch", hide_index=True)
    else:
        st.caption("当前搜索下无考生记录")

    # 单条记录模块雷达图（个人单次考试）—— 自绘 SVG，pills 切换不可手动输入，切换带过渡动画
    st.markdown("---")
    st.subheader("📊 个人单次考试雷达图")
    _radar_html = _single_radar_animated(records)
    if _radar_html:
        components.html(_radar_html, height=480)
    else:
        st.caption("暂无记录")

    # 错题明细（二级菜单：按时间排列每次考试，点开显示该次错题）
    with st.expander("📋 查看所有错题明细"):
        has_any = False
        for r in records:
            wrong = r.get("wrong_details") or []
            if not wrong:
                continue
            has_any = True
            ts = (r["created_at"] or "")[:16]
            title = (f"📅 {ts} ｜ {r['name'] or '未命名'} ｜ "
                     f"总分 {r['total_score']} ｜ 错 {len(wrong)} 题")
            with st.expander(title, expanded=False):
                st.dataframe(pd.DataFrame(wrong), width="stretch", hide_index=True)
        if not has_any:
            st.caption("暂无错题记录")


# --------------------------------------------------------------------- #
# Tab4 配置管理
# --------------------------------------------------------------------- #
def render_config():
    st.subheader("配置管理（标准答案 / 模块分值）")
    if not current_user():
        st.warning("⚠️ 请先登录（左侧「用户工作区」）后再配置你的标准答案与模块分值。"
                   "配置按用户完全隔离，未登录不可查看/修改。")
        return

    st.markdown("#### 从文件导入标准答案（Excel / Word / PDF / TXT）")
    up = st.file_uploader(
        "上传答案文件，解析后写入 configs/standard_answers.json 并设为默认",
        type=["xlsx", "xls", "docx", "pdf", "txt", "csv"], key="ans_upload_config")
    if up is not None:
        digest = hashlib.md5(up.getvalue()).hexdigest()
        if st.session_state.get("session_answers_digest") != digest:
            load_answers_from_upload(up)
        if st.session_state.get("session_answers"):
            prev = list(sorted(st.session_state["session_answers"].items()))[:10]
            st.caption("解析预览（前10题）：" + "  ".join(f"{k}.{v}" for k, v in prev) + " …")
            if st.button("✅ 确认导入并设为默认答案", key="import_ans"):
                data = st.session_state["session_answers"]
                save_standard_answers({str(k): v for k, v in data.items()},
                                      user=current_user())
                st.session_state["standard"] = data
                st.success(f"已导入 {len(data)} 题"
                           + ("（已写入你的工作区）" if current_user() else "（已写入全局 configs）"))

    st.markdown("#### 标准答案 JSON")
    std_text = st.text_area(
        "标准答案：{\"题号\": \"A/B/C/D\"}，覆盖全部题目",
        value=json.dumps({k: st.session_state["standard"][k] for k in
                          sorted(st.session_state["standard"], key=lambda x: int(x))},
                         ensure_ascii=False, indent=1),
        height=240, key="std_text")
    if st.button("保存标准答案", key="save_std"):
        try:
            data = json.loads(std_text)
            cleaned = {str(k): str(v).strip().upper() for k, v in data.items()
                       if str(v).strip().upper() in OPTIONS}
            save_standard_answers(cleaned, user=current_user())
            st.session_state["standard"] = cleaned
            st.success(f"已保存 {len(cleaned)} 道标准答案"
                       + ("（你的工作区）" if current_user() else "（全局）"))
        except json.JSONDecodeError:
            st.error("JSON 格式错误，请检查")

    st.markdown("#### 模块分值配置")
    st.caption("不同地区/试卷类型题量不同：判卷时总题量默认按答案文件题数自动判定"
               "（侧边栏可手动指定）。若总题量超过最后一模块的「结束题号」，"
               "系统会自动扩展最后一模块覆盖到该题量计分；如需精确划分请在此编辑起止与权重。")
    mods = st.session_state["modules"]
    mod_df = pd.DataFrame([{
        "模块": k, "起始题号": v["start"], "结束题号": v["end"], "单题权重": v["weight"],
    } for k, v in mods.items()])
    edited = st.data_editor(mod_df, num_rows="dynamic", width="stretch",
                            key="mod_editor")
    if st.button("保存模块配置"):
        new_mods = {}
        for _, r in edited.iterrows():
            try:
                new_mods[str(r["模块"])] = {
                    "start": int(r["起始题号"]), "end": int(r["结束题号"]),
                    "weight": float(r["单题权重"]),
                }
            except (ValueError, TypeError):
                continue
        if new_mods:
            save_module_weights(new_mods, user=current_user())
            st.session_state["modules"] = new_mods
            st.success("模块配置已保存"
                       + ("（你的工作区）" if current_user() else "（全局）"))


# --------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------- #
def main():
    init_state()
    render_sidebar()
    # 全局样式：统一按钮圆角点击效果（去掉直角 outline/focus 阴影）
    st.markdown("""<style>
    button:focus,button:active{outline:none!important;box-shadow:none!important;}
    button{-webkit-tap-highlight-color:transparent;}
    .stButton button:active{transform:scale(0.97);}
    </style>""", unsafe_allow_html=True)
    st.title("📝 公务员行测答题卡扫描评分系统")
    st.caption("Python + OpenCV 识别 / Streamlit 界面 · 支持纵向S型排列 · 答案自动定题量，130/135/140 题自适应判卷")
    if not current_user():
        # 未登录：全部功能封锁（上传答案/答题卡评分、历史、配置均不可用），避免数据混入全局
        st.info("👈 请先在左侧「用户工作区」输入用户名登录（首次输入即创建你的工作区）。")
        st.caption("登录后才能使用：上传答案与答题卡评分、保存历史记录、配置模块分值。"
                   "每个用户的数据完全隔离，未登录不显示、不写入任何数据。")
        return
    tab1, tab2, tab3, tab4 = st.tabs(["🧾 单张识别", "📚 批量识别", "📊 历史统计", "⚙️ 配置管理"])
    with tab1:
        render_single()
    with tab2:
        render_batch()
    with tab3:
        render_history()
    with tab4:
        render_config()


if __name__ == "__main__":
    main()

