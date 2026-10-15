#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPMQA RAG 架构图生成器（纯标准库，无需任何第三方依赖）

一次运行同时产出三个坐标完全一致的图源文件：
    rag-architecture.svg     —— 矢量图，浏览器 / VS Code 直接看
    rag-architecture.drawio  —— draw.io 可打开、可手工精修、可导出高清 PNG
    rag-architecture.html    —— 内嵌 SVG，带「下载 PNG」按钮，双击即看

自检（本机无渲染器 / 无中文字体，无法肉眼验证，因此把画图常见的错都写成检查）：
    * 元素越界      方框 / 阶段带是否落在画布内
    * 文字溢出      按字宽估算每行文字是否超出方框
    * 方框重叠      方框两两相交
    * 连线穿框      正交折线是否从方框内部穿过
    * 标签压框      连线标签白底是否盖到方框
    * 注解越界/压框 文字注解是否超画布或被遮挡
    * XML 良构      生成的 SVG / drawio 用 ElementTree 解析一遍

用法:
    python3 gen_architecture.py            # 自检 + 重写 svg / drawio / html
    python3 gen_architecture.py --check    # 只自检，不写文件

图的内容与改图说明见同目录 README.md。
"""

import os
import sys
import argparse
from xml.sax.saxutils import escape

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 画布 / 字体
W, H = 1960, 1580

# 中文优先走 Windows 侧字体（本机是 WSL，Linux 侧无 CJK 字体，
# 但渲染发生在 Windows 浏览器 / VS Code 预览，微软雅黑必然存在）
FONT = ("'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',"
        "'Source Han Sans SC',SimHei,sans-serif")

INK = "#1f2933"      # 主文字
INK2 = "#4b5563"     # 次级文字
INK3 = "#8a94a6"     # 弱化文字

# 四阶段配色（低饱和）
C_OFF = dict(fill="#f3f7fd", stroke="#c3d8ee", accent="#2b5c8a", soft="#ffffff")
C_ONL = dict(fill="#fdf6ec", stroke="#efd6b4", accent="#a5601a", soft="#fffdf9")
C_EVL = dict(fill="#f5f1fc", stroke="#dccff0", accent="#6b46a8", soft="#ffffff")
C_MDL = dict(fill="#f1faf4", stroke="#c6e5d2", accent="#2b7a4b", soft="#ffffff")


def tw(s, fs):
    """估算文字像素宽度：CJK / 全角符号按 1.0em，其余按 0.58em。

    微软雅黑里汉字与全角标点的 advance 正好是 1.0em，拉丁字母平均约 0.55em，
    所以 0.58 是偏保守（偏大）的估计 —— 宁可行长报警，不要肉眼才发现溢出。
    偏大的估计对「文字溢出」检查是安全的。
    """
    w = 0.0
    for ch in s:
        o = ord(ch)
        if (o > 0x2E80                      # CJK 部首 / 汉字 / 假名 / 兼容区
                or 0x2460 <= o <= 0x24FF    # ①②③ 圈号
                or 0x2000 <= o <= 0x2BFF):  # 通用标点 / 箭头 / 数学符号 / 制表符
            w += fs
        else:
            w += fs * 0.58
    return w


# ---------------------------------------------------------------- 结构定义
BANDS = []
BOXES = []
ARROWS = []
NOTES = []


def band(bid, x, y, w, h, title, caption=None, pal=None):
    BANDS.append(dict(id=bid, x=x, y=y, w=w, h=h, title=title,
                      caption=caption, pal=pal or C_OFF))


def box(bid, x, y, w, h, title, lines=None, pal=None, fs_title=14.5, fs=12.0,
        align="middle"):
    BOXES.append(dict(id=bid, x=x, y=y, w=w, h=h, title=title,
                      lines=lines or [], pal=pal or C_OFF,
                      fs_title=fs_title, fs=fs, align=align))


def arrow(aid, pts, dashed=False, label=None, label_xy=None, color=None):
    ARROWS.append(dict(id=aid, pts=list(pts), dashed=dashed, label=label,
                       label_xy=label_xy, color=color or "#6b7280"))


def note(nid, x, y, text, fs=12.0, color=None, anchor="start"):
    NOTES.append(dict(id=nid, x=x, y=y, text=text, fs=fs,
                      color=color or INK3, anchor=anchor))


# ================================================================ 版式数据
# 布局原则：
#   左侧三横带（离线 / 在线 / 评估） + 右侧一竖列（模型部署）
#   所有跨带连线走带与带之间的空隙（x=1520~1560 的竖直水槽 + y=500~540 的横向水槽）

# ---- 画布标题 ----
TITLE = "MPMQA RAG 系统架构"
SUBTITLE = ("覆盖 切片 → 入库 → 模型部署 → 后端 → 前端 → 评估 全流程　|　"
            "多数据集隔离：GWM / MDM（--dataset 驱动 collection / 路径 / 端口）")

# ---- 四个阶段带 ----
band("b_off", 40, 140, 1480, 360,
     "① 离线阶段 · 切片 + 入库", None, C_OFF)
band("b_onl", 40, 540, 1480, 660,
     "② 在线服务 · Streamlit 前端 + FastAPI 后端", None, C_ONL)
band("b_evl", 40, 1240, 1480, 240,
     "③ 评估 · 跑批测试（Ragas）", None, C_EVL)
band("b_mdl", 1560, 140, 360, 1300,
     "④ 模型部署 · deploy/*.sh", "vLLM · OpenAI 兼容 API", C_MDL)

# ---- ① 离线 ----
box("docs", 80, 220, 230, 80,
    "data/docs/{DS}/*.docx", ["原始 Word 文档"], C_OFF)
box("parser", 370, 220, 270, 80,
    "docx_parser.py", ["按 Heading 层级切分 + 抽图"], C_OFF)
box("chunks", 700, 220, 240, 80,
    "docs/{DS}_json/", ["chunks（含 title_path）"], C_OFF)
box("indexer", 990, 220, 240, 80,
    "indexer.py", ["批量 upsert（batch=32）"], C_OFF)
box("chroma", 1290, 220, 190, 80,
    "ChromaDB", ["collection={DS}_docs", "HNSW cosine"], C_OFF, 13.5, 11.0)
box("images", 80, 350, 230, 80,
    "data/images/{DS}/", ["img_*.png（抽出的图）"], C_OFF)

# ---- ② 在线 ----
box("fe", 80, 690, 300, 200,
    "Streamlit 前端",
    ["frontend/app.py", ":8501 GWM · :8502 MDM",
     "session_id(UUID) · 历史渲染", "检索模式 vector/bm25/hybrid",
     "[ref:uid] → 图片内联放大", "requests.post → /ask"], C_ONL, 14.0, 11.5)
box("api", 420, 640, 1080, 540,
    "FastAPI 后端 · backend/api.py",
    ["SessionManager：内存 · 滑动窗口 3 条 · TTL 1800s · 后台清理线程 · 默认 session=default"],
    C_ONL, 14.5, 12.0, align="top")
box("c1", 450, 735, 230, 130,
    "① intent_chain", ["意图判别", "thanks / chitchat", "→ 直接回复（短路）"],
    C_ONL, 13.5, 11.5)
box("c2", 710, 735, 230, 130,
    "② rewrite_chain", ["多轮指代消解", "剥离 'rewritten query:'"],
    C_ONL, 13.5, 11.5)
box("c3", 970, 735, 230, 130,
    "③ Retriever", ["mode + top_k=10", "vector: Chroma ≤1.0",
                    "bm25: rank_bm25", "hybrid: RRF(k=60)"],
    C_ONL, 13.5, 11.5)
box("c4", 1230, 735, 230, 130,
    "④ rag_chain", ["system_prompt", "+ history + context", "→ 含 [ref:uid] 答案"],
    C_ONL, 13.5, 11.5)
box("c5", 1230, 1000, 230, 120,
    "⑤ 后处理", ["normalize_ref_position", "dedup_refs", "strip_refs（入历史）"],
    C_ONL, 13.5, 11.0)

# ---- ③ 评估 ----
box("v1", 80, 1300, 260, 110,
    "data/test/{DS}/*.xlsx", ["测试集 (query, answer)"], C_EVL, 13.5, 11.5)
box("v2", 400, 1300, 280, 110,
    "eval_ragas.py", ["批量 POST /ask", "收集 answer + contexts"], C_EVL, 13.5, 11.5)
box("v3", 740, 1300, 300, 110,
    "Ragas 指标", ["faithfulness · answer_relevancy",
                   "context_precision · context_recall"], C_EVL, 13.5, 11.5)
box("v4", 1100, 1300, 380, 110,
    "ragas_results/{DS}/", ["{doc}_{mode}_ragas_result.xlsx"], C_EVL, 13.5, 11.5)

# ---- ④ 模型部署（右列） ----
box("chat", 1580, 240, 320, 150,
    "vLLM chat", [":8000", "Qwen3.5-4B-AWQ",
                  "gpu-util 0.65 · max-len 8192",
                  "num-seqs 8 · enforce-eager"], C_MDL, 14.5, 11.5)
box("emb", 1580, 440, 320, 150,
    "vLLM embedding", [":8001", "Qwen3-Embedding-0.6B",
                       "gpu-util 0.25 · max-len 800", "num-seqs 8"], C_MDL, 14.5, 11.5)
box("dep", 1580, 660, 320, 190,
    "部署脚本",
    ["deploy_all.sh", " ├ deploy_chat.sh → :8000",
     " └ deploy_emb.sh → :8001", "common.sh：conda rag · CUDA · 代理",
     "wait_ready()：轮询 /health 60×3s"], C_MDL, 13.5, 11.0)
box("ports", 1580, 890, 320, 190,
    "端口速查",
    ["8000/8001  vLLM chat / embedding", "8002  后端 GWM · GWM_docs",
     "8003  后端 MDM · MDM_docs", "8501/8502  前端 GWM / MDM"],
    C_MDL, 13.5, 11.0)
box("logs", 1580, 1120, 320, 130,
    "日志",
    ["$LOG_DIR/vllm-chat.log", "$LOG_DIR/vllm-emb.log", "$LOG_DIR = $HOME/logs"],
    C_MDL, 13.5, 11.0)
box("redo", 1580, 1290, 320, 130,
    "重建索引",
    ["python ingestion/docx_parser.py", "python ingestion/indexer.py"],
    C_MDL, 13.5, 10.5)

# ---- 离线内部连线 ----
arrow("a1", [(310, 260), (370, 260)], False, "读取", (340, 246))
arrow("a2", [(640, 260), (700, 260)], True, "chunks", (670, 246))
arrow("a3", [(505, 300), (505, 325), (195, 325), (195, 350)], True,
      "抽图落盘", (505, 340))
arrow("a4", [(940, 260), (990, 260)], True, "切片", (965, 246))
arrow("a5", [(1230, 260), (1290, 260)], False, "upsert", (1260, 246))
# indexer -> embedding 服务（走画布右侧竖直水槽 x≈1535）
arrow("a6", [(1110, 220), (1110, 180), (1535, 180), (1535, 500), (1580, 500)],
      True, "indexer 调 /v1/embeddings", (1130, 172))
# 抽出的图片 -> 前端（前端按 [ref:uid] 取图）
arrow("a7", [(195, 430), (195, 690)], True, "图片资源", (195, 520))

# ---- 在线内部连线 ----
arrow("a8", [(380, 800), (450, 800)], False, "POST /ask", (415, 786))
arrow("a9", [(680, 800), (710, 800)], False)
arrow("a10", [(940, 800), (970, 800)], False)
arrow("a11", [(1200, 800), (1230, 800)], False)
arrow("a12", [(1345, 865), (1345, 1000)], False, "answer", (1345, 930))
arrow("a13", [(1230, 1060), (230, 1060), (230, 890)], False,
      "answer + sources[]", (700, 1048))
# 在线 -> chat（intent / rewrite / rag 三个 chain 汇总成一条调用线）
arrow("a14", [(880, 735), (880, 520), (1545, 520), (1545, 315), (1580, 315)],
      False, "ChatOpenAI", (900, 520))
# 在线 -> embedding（query 向量化）
arrow("a15", [(1085, 735), (1085, 505), (1515, 505), (1515, 545), (1580, 545)],
      False, "OpenAIEmbeddings", (1380, 505))

# ---- 评估连线 ----
arrow("a16", [(340, 1355), (400, 1355)], False, "读取", (370, 1341))
arrow("a17", [(680, 1355), (740, 1355)], False)
arrow("a18", [(1040, 1355), (1100, 1355)], False, "写出", (1070, 1341))
arrow("a19", [(540, 1300), (540, 1185)], False, "POST /ask", (640, 1232))
# 评估 -> embedding（Ragas 的 answer_relevancy 等复用同一对服务）
arrow("a20", [(1480, 1355), (1540, 1355), (1540, 620), (1700, 620), (1700, 590)],
      True)

# ---- 文字注解 ----
note("n_off", 80, 470,
     "docx_parser：仅从 Heading 1 起进入正文；遇新标题或图片边界 flush；标题回填进切片正文",
     11.5)
note("n_evl", 80, 1450,
     "评估复用 :8000 / :8001 同一对 vLLM 服务（不新建实例）　|　"
     "run_eval.sh 按 DATASETS × MODES 组合遍历", 11.5)
note("n_legend", 1400, 52, "图例", 13.0, INK2)
note("n_leg1", 1400, 76, "───  同步调用（HTTP / 函数）", 11.5)
note("n_leg2", 1400, 98, "┄┄┄  文件落盘 / 索引数据流", 11.5)
note("n_foot", 64, 1530,
     "图源：docs/diagrams/gen_architecture.py（一次生成 SVG + drawio，坐标一致）"
     "　|　架构说明：docs/architecture.md", 11.5)


# ================================================================ 内部工具
def box_rows(b):
    """返回 (行文本列表, 行高, 首行基线 y)"""
    n = len(b["lines"])
    lh = min(19.0, max(12.5, (b["h"] - 34) / (n + 1.35)))
    rows = [b["title"]] + list(b["lines"])
    if b["align"] == "top":
        first = b["y"] + 20 + lh * 0.8
    else:
        first = b["y"] + (b["h"] - (n + 1) * lh) / 2 + lh * 0.8
    return rows, lh, first


def seg_mid(pts):
    """取最长一段的中点，用于放置连线标签"""
    best, bi = -1.0, 0
    for i in range(len(pts) - 1):
        d = (pts[i][0] - pts[i + 1][0]) ** 2 + (pts[i][1] - pts[i + 1][1]) ** 2
        if d > best:
            best, bi = d, i
    (x1, y1), (x2, y2) = pts[bi], pts[bi + 1]
    return (x1 + x2) / 2, (y1 + y2) / 2


def label_anchor(a):
    if a["label_xy"]:
        return a["label_xy"]
    return seg_mid(a["pts"])


# ================================================================ 自检
def validate():
    warn = []

    def rect_ok(tag, x, y, w, h):
        if x < 0 or y < 0 or x + w > W or y + h > H:
            warn.append(f"[越界] {tag}: ({x},{y},{w},{h}) 超出画布 {W}x{H}")

    for b in BANDS:
        rect_ok("band " + b["id"], b["x"], b["y"], b["w"], b["h"])
    for b in BOXES:
        rect_ok("box " + b["id"], b["x"], b["y"], b["w"], b["h"])
        if b["id"] == "api":
            continue  # 容器框，不校验文字宽度
        rows, _, _ = box_rows(b)
        for r in rows:
            fs = b["fs_title"] if r is rows[0] else b["fs"]
            need = tw(r, fs)
            if need > b["w"] - 16:
                warn.append(
                    f"[文字溢出] box {b['id']} ({b['w']}px 宽): "
                    f"“{r}” 需 {need:.0f}px")
    # 盒子两两重叠检测
    for i in range(len(BOXES)):
        for j in range(i + 1, len(BOXES)):
            a, c = BOXES[i], BOXES[j]
            if a["id"] == "api" or c["id"] == "api":
                continue  # 容器框允许包含子框
            if (a["x"] < c["x"] + c["w"] and c["x"] < a["x"] + a["w"]
                    and a["y"] < c["y"] + c["h"] and c["y"] < a["y"] + a["h"]):
                warn.append(f"[盒子重叠] {a['id']} x {c['id']}")
    # 连线线段是否穿过方框内部（正交折线，只处理水平/垂直段）
    def seg_hits(p1, p2, b):
        (x1, y1), (x2, y2) = p1, p2
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        if x1 == x2:
            if not (bx < x1 < bx + bw):
                return False
            lo, hi = min(y1, y2), max(y1, y2)
            return lo < by + bh and by < hi
        if y1 == y2:
            if not (by < y1 < by + bh):
                return False
            lo, hi = min(x1, x2), max(x1, x2)
            return lo < bx + bw and bx < hi
        return False

    for a in ARROWS:
        for k in range(len(a["pts"]) - 1):
            for b in BOXES:
                if b["id"] == "api":
                    continue
                if seg_hits(a["pts"][k], a["pts"][k + 1], b):
                    warn.append(
                        f"[线穿盒] arrow {a['id']} 第{k + 1}段穿过 box {b['id']}")

    # 连线标签是否落在某个盒子上（会被盒子遮住）
    for a in ARROWS:
        if not a["label"]:
            continue
        lx, ly = label_anchor(a)
        hw = tw(a["label"], 11.5) / 2 + 3
        for b in BOXES:
            if b["id"] == "api":
                continue  # 容器框，标签落在其空白区是正常的
            if (b["x"] < lx + hw and lx - hw < b["x"] + b["w"]
                    and b["y"] < ly + 8 and ly - 8 < b["y"] + b["h"]):
                warn.append(
                    f"[标签压框] arrow {a['id']} 标签“{a['label']}” "
                    f"落在 box {b['id']} 上")
                break
    # 注解是否越界 / 压在方框上
    for n in NOTES:
        nw = tw(n["text"], n["fs"])
        x2 = n["x"] + nw if n["anchor"] == "start" else n["x"] + nw / 2
        x1 = n["x"] if n["anchor"] == "start" else n["x"] - nw / 2
        if x1 < 0 or x2 > W or n["y"] < 0 or n["y"] > H:
            warn.append(f"[注解越界] note {n['id']}: x {x1:.0f}~{x2:.0f} y={n['y']}")
        for b in BOXES:
            if b["id"] == "api":
                continue
            if (b["x"] < x2 and x1 < b["x"] + b["w"]
                    and b["y"] < n["y"] + 4 and n["y"] - n["fs"] < b["y"] + b["h"]):
                warn.append(f"[注解压框] note {n['id']} 落在 box {b['id']} 上")
                break

    return warn


# ================================================================ SVG 输出
def to_svg(xml_decl=True):
    o = []
    a = o.append
    if xml_decl:
        a('<?xml version="1.0" encoding="UTF-8"?>')
    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" font-family="{escape(FONT, {chr(34): "&quot;"})}">')
    a('<defs>')
    a('<marker id="mk" viewBox="0 0 10 10" refX="9.4" refY="5" '
      'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
      '<path d="M 0 0.8 L 9.6 5 L 0 9.2 z" fill="#6b7280"/></marker>')
    a('</defs>')
    a(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # 标题区
    a(f'<text x="64" y="72" font-size="30" font-weight="bold" fill="#111827">'
      f'{escape(TITLE)}</text>')
    a(f'<text x="64" y="106" font-size="14" fill="{INK2}">'
      f'{escape(SUBTITLE)}</text>')

    # 阶段带
    for b in BANDS:
        p = b["pal"]
        a(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" '
          f'rx="14" fill="{p["fill"]}" stroke="{p["stroke"]}" stroke-width="1.2"/>')
        a(f'<text x="{b["x"] + 24}" y="{b["y"] + 34}" font-size="14" '
          f'font-weight="bold" fill="{p["accent"]}">{escape(b["title"])}</text>')
        if b["caption"]:
            a(f'<text x="{b["x"] + 24}" y="{b["y"] + 56}" font-size="11.5" '
              f'fill="{INK2}">{escape(b["caption"])}</text>')

    # 连线
    for ar in ARROWS:
        pts = " ".join(f"{x},{y}" for x, y in ar["pts"])
        dash = ' stroke-dasharray="7 5"' if ar["dashed"] else ""
        a(f'<polyline points="{pts}" fill="none" stroke="{ar["color"]}" '
          f'stroke-width="1.6"{dash} marker-end="url(#mk)"/>')

    # 方框
    for b in BOXES:
        p = b["pal"]
        a(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" '
          f'rx="9" fill="{p["soft"]}" stroke="{p["accent"]}" stroke-width="1.3"/>')
        rows, lh, first = box_rows(b)
        cx = b["x"] + b["w"] / 2
        for i, r in enumerate(rows):
            y = first + i * lh
            if i == 0:
                a(f'<text x="{cx:.0f}" y="{y:.0f}" font-size="{b["fs_title"]}" '
                  f'font-weight="bold" fill="{p["accent"]}" text-anchor="middle">'
                  f'{escape(r)}</text>')
            else:
                a(f'<text x="{cx:.0f}" y="{y:.0f}" font-size="{b["fs"]}" '
                  f'fill="{INK2}" text-anchor="middle">{escape(r)}</text>')

    # 连线标签（画在方框之上，白底遮线）
    for ar in ARROWS:
        if not ar["label"]:
            continue
        lx, ly = label_anchor(ar)
        wd = tw(ar["label"], 11.5) + 6
        a(f'<rect x="{lx - wd / 2:.1f}" y="{ly - 9:.1f}" width="{wd:.1f}" '
          f'height="18" rx="4" fill="#ffffff" fill-opacity="0.92"/>')
        a(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" font-size="11.5" fill="{INK2}" '
          f'text-anchor="middle">{escape(ar["label"])}</text>')

    # 注解
    for n in NOTES:
        a(f'<text x="{n["x"]}" y="{n["y"]}" font-size="{n["fs"]}" '
          f'fill="{n["color"]}" text-anchor="{n["anchor"]}">'
          f'{escape(n["text"])}</text>')

    a('</svg>')
    return "\n".join(o) + "\n"


# ================================================================ drawio 输出
def xa(s):
    return escape(str(s), {'"': "&quot;"})


def to_drawio():
    o = []
    a = o.append
    a('<mxfile host="app.diagrams.net" agent="gen_architecture.py" '
      'type="device">')
    a(f'  <diagram id="mpmqa-rag" name="MPMQA RAG 架构">')
    a(f'    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" '
      f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
      f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">')
    a('      <root>')
    a('        <mxCell id="0"/>')
    a('        <mxCell id="1" parent="0"/>')

    # 1) 阶段带：纯矩形，标题/副标题另用文本顶点（与 SVG 版式一致）
    for b in BANDS:
        p = b["pal"]
        st = (f'rounded=1;whiteSpace=wrap;html=1;fillColor={p["fill"]};'
              f'strokeColor={p["stroke"]};strokeWidth=1.2;movable=1;'
              f'resizable=1;fontFamily=Microsoft YaHei;')
        a(f'        <mxCell id="{xa(b["id"])}" value="" style="{xa(st)}" '
          f'vertex="1" parent="1">')
        a(f'          <mxGeometry x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" '
          f'height="{b["h"]}" as="geometry"/>')
        a('        </mxCell>')
        tst = (f'text;html=1;align=left;verticalAlign=top;fontColor={p["accent"]};'
               f'fontSize=14;fontStyle=1;strokeColor=none;fillColor=none;'
               f'fontFamily=Microsoft YaHei;')
        a(f'        <mxCell id="{xa(b["id"] + "_t")}" value="{xa(b["title"])}" '
          f'style="{xa(tst)}" vertex="1" parent="1">')
        a(f'          <mxGeometry x="{b["x"] + 16}" y="{b["y"] + 16}" '
          f'width="{b["w"] - 32}" height="24" as="geometry"/>')
        a('        </mxCell>')
        if b["caption"]:
            cst = (f'text;html=1;align=left;verticalAlign=top;fontColor={INK2};'
                   f'fontSize=11;strokeColor=none;fillColor=none;'
                   f'fontFamily=Microsoft YaHei;')
            a(f'        <mxCell id="{xa(b["id"] + "_c")}" '
              f'value="{xa(b["caption"])}" style="{xa(cst)}" vertex="1" parent="1">')
            a(f'          <mxGeometry x="{b["x"] + 16}" y="{b["y"] + 40}" '
              f'width="{b["w"] - 32}" height="20" as="geometry"/>')
            a('        </mxCell>')

    # 2) 连线（带折点的正交折线；标签单独做文本顶点，保证与 SVG 位置一致）
    for ar in ARROWS:
        pts = ar["pts"]
        st = (f'edgeStyle=none;html=1;rounded=0;jumpStyle=none;'
              f'strokeColor={ar["color"]};strokeWidth=1.6;'
              f'dashed={1 if ar["dashed"] else 0};endArrow=block;endFill=1;'
              f'endSize=6;fontFamily=Microsoft YaHei;')
        a(f'        <mxCell id="{xa(ar["id"])}" value="" style="{xa(st)}" '
          f'edge="1" parent="1">')
        a('          <mxGeometry relative="1" as="geometry">')
        a(f'            <mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>')
        a(f'            <mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>')
        if len(pts) > 2:
            a('            <Array as="points">')
            for x, y in pts[1:-1]:
                a(f'              <mxPoint x="{x}" y="{y}"/>')
            a('            </Array>')
        a('          </mxGeometry>')
        a('        </mxCell>')

    # 3) 方框
    for b in BOXES:
        p = b["pal"]
        parts = [f'<font style="font-size:{b["fs_title"]}px"><b>{b["title"]}</b></font>']
        for l in b["lines"]:
            parts.append(f'<font style="font-size:{b["fs"]}px">{l}</font>')
        val = "<br>".join(parts)
        st = (f'rounded=1;whiteSpace=wrap;html=1;fillColor={p["soft"]};'
              f'strokeColor={p["accent"]};strokeWidth=1.3;fontColor={INK};'
              f'fontFamily=Microsoft YaHei;verticalAlign=middle;')
        a(f'        <mxCell id="{xa(b["id"])}" value="{xa(val)}" '
          f'style="{xa(st)}" vertex="1" parent="1">')
        a(f'          <mxGeometry x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" '
          f'height="{b["h"]}" as="geometry"/>')
        a('        </mxCell>')

    # 4) 连线标签
    for ar in ARROWS:
        if not ar["label"]:
            continue
        lx, ly = label_anchor(ar)
        wd = tw(ar["label"], 11.5) + 6
        lst = (f'text;html=1;align=center;verticalAlign=middle;fontColor={INK2};'
               f'fontSize=11;strokeColor=none;fillColor=#ffffff;'
               f'fontFamily=Microsoft YaHei;')
        a(f'        <mxCell id="{xa(ar["id"] + "_l")}" value="{xa(ar["label"])}" '
          f'style="{xa(lst)}" vertex="1" parent="1">')
        a(f'          <mxGeometry x="{lx - wd / 2:.0f}" y="{ly - 9:.0f}" '
          f'width="{wd:.0f}" height="18" as="geometry"/>')
        a('        </mxCell>')

    # 5) 纯文字注解
    for n in NOTES:
        nst = (f'text;html=1;align={n["anchor"]};verticalAlign=middle;'
               f'fontColor={n["color"]};fontSize={n["fs"]};strokeColor=none;'
               f'fillColor=none;fontFamily=Microsoft YaHei;')
        a(f'        <mxCell id="{xa(n["id"])}" value="{xa(n["text"])}" '
          f'style="{xa(nst)}" vertex="1" parent="1">')
        a(f'          <mxGeometry x="{n["x"]}" y="{n["y"] - 12}" '
          f'width="{tw(n["text"], n["fs"]) + 8:.0f}" height="20" as="geometry"/>')
        a('        </mxCell>')

    a('      </root>')
    a('    </mxGraphModel>')
    a('  </diagram>')
    a('</mxfile>')
    return "\n".join(o) + "\n"


# ================================================================ HTML 输出
HTML_TPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MPMQA RAG 系统架构图</title>
<style>
  html, body { margin: 0; padding: 0; background: #f6f7f9;
               font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; }
  header { position: sticky; top: 0; z-index: 10; display: flex;
           align-items: center; gap: 12px; flex-wrap: wrap;
           padding: 10px 18px; background: #ffffff;
           border-bottom: 1px solid #e3e6ea; }
  header .t { font-weight: 700; color: #1f2933; margin-right: 8px; }
  header label { color: #4b5563; font-size: 13px; }
  select, button { font: inherit; font-size: 13px; padding: 5px 12px;
                   border: 1px solid #d0d5dc; border-radius: 6px;
                   background: #fff; color: #1f2933; cursor: pointer; }
  button { background: #2b5c8a; border-color: #2b5c8a; color: #fff;
           font-weight: 600; }
  button:hover { background: #244d75; }
  #msg { font-size: 12.5px; color: #2b7a4b; }
  #msg.err { color: #b4321f; }
  #stage { padding: 18px; }
  #stage svg { width: 100%; height: auto; display: block;
               background: #fff; border: 1px solid #e3e6ea;
               border-radius: 10px; box-shadow: 0 1px 3px rgba(16,24,40,.06); }
  footer { padding: 0 18px 20px; font-size: 12px; color: #8a94a6; }
  code { background: #eef1f5; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<header>
  <span class="t">MPMQA RAG 系统架构</span>
  <label for="scale">导出倍率</label>
  <select id="scale">
    <option value="1">1x（1960 px）</option>
    <option value="2" selected>2x（3920 px）</option>
    <option value="3">3x（5880 px）</option>
  </select>
  <button id="dl">下载 PNG</button>
  <span id="msg">提示：若 PNG 文字异常，请改用 draw.io 导出（见 README.md）</span>
</header>

<div id="stage">
__SVG__
</div>

<footer>
  图源：<code>docs/diagrams/gen_architecture.py</code>（一次生成 SVG + drawio，坐标一致）　|　
  架构说明：<code>docs/architecture.md</code>
</footer>

<script>
(function () {
  var msg = document.getElementById('msg');
  function say(t, isErr) {
    msg.textContent = t;
    msg.className = isErr ? 'err' : '';
  }
  document.getElementById('dl').addEventListener('click', function () {
    var svg = document.querySelector('#stage svg');
    if (!svg) { say('未找到 SVG 节点', true); return; }
    var scale = parseFloat(document.getElementById('scale').value) || 2;
    var vb = svg.viewBox && svg.viewBox.baseVal;
    var w = (vb && vb.width) || svg.width.baseVal.value;
    var h = (vb && vb.height) || svg.height.baseVal.value;

    var clone = svg.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    var src = new XMLSerializer().serializeToString(clone);
    var url = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(src);

    var img = new Image();
    img.onload = function () {
      var cv = document.createElement('canvas');
      cv.width = Math.round(w * scale);
      cv.height = Math.round(h * scale);
      var ctx = cv.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.drawImage(img, 0, 0, w, h);
      cv.toBlob(function (blob) {
        if (!blob) { say('导出失败，请改用 draw.io', true); return; }
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'rag-architecture-' + scale + 'x.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
        say('已导出 rag-architecture-' + scale + 'x.png');
      }, 'image/png');
    };
    img.onerror = function () {
      say('SVG 渲染失败，请改用 draw.io 导出 PNG（见 README.md）', true);
    };
    img.src = url;
  });
})();
</script>
</body>
</html>
"""


def to_html(svg):
    return HTML_TPL.replace("__SVG__", svg)


# ================================================================ 入口
def main():
    ap = argparse.ArgumentParser(description="生成 MPMQA RAG 架构图（SVG + drawio + HTML）")
    ap.add_argument("--check", action="store_true", help="只自检，不写文件")
    args = ap.parse_args()

    warns = validate()
    if warns:
        print(f"⚠ 自检发现 {len(warns)} 个问题：")
        for w in warns:
            print("   -", w)
    else:
        print("✅ 自检通过：元素未越界 / 文字未溢出 / 方框无重叠 / "
              "连线未穿框 / 标签无压框")

    if args.check:
        return 1 if warns else 0

    targets = [
        ("rag-architecture.svg", to_svg(True)),
        ("rag-architecture.drawio", to_drawio()),
        ("rag-architecture.html", to_html(to_svg(False))),
    ]

    import xml.etree.ElementTree as ET
    for name, content in targets:
        if name.endswith(".html"):
            continue
        try:
            ET.fromstring(content)
            print(f"   XML 良构 OK  {name}")
        except ET.ParseError as e:
            print(f"❌ {name} XML 解析失败: {e}")
            return 1

    for name, content in targets:
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"已写入 {path}  ({len(content)} bytes)")

    print(f"\n版式：{W}x{H}　盒子 {len(BOXES)} 个　连线 {len(ARROWS)} 条　"
          f"阶段带 {len(BANDS)} 个　注解 {len(NOTES)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())




