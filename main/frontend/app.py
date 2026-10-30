import argparse
import uuid
import re
import os
import json
import uuid

import streamlit as st
from streamlit_image_zoom import image_zoom
from PIL import Image
import requests

from config import FrontendConfig, MODE_MAP, PROXIES


# ---------- argparse ----------
ap = argparse.ArgumentParser()
ap.add_argument("--api-url", default="http://127.0.0.1:8002/ask")
ap.add_argument("--image-base", default="./data/images/GWM/")
ap.add_argument("--title-suffix", default="")
args, _ = ap.parse_known_args()


def _derive_endpoints(api_url: str):
    """
    兼容两种写法：
      --api-url http://127.0.0.1:8002          → base
      --api-url http://127.0.0.1:8002/ask      → 完整旧路径
    返回 (base, ask_url, stream_url, sources_url)
    """
    url = api_url.rstrip("/")
    if url.endswith("/ask"):
        base = url[:-len("/ask")]
    else:
        base = url
    return (
        base,
        f"{base}/ask",
        f"{base}/ask/stream",
        f"{base}/sources",
    )


_base, ASK_URL, STREAM_URL, SOURCES_URL = _derive_endpoints(args.api_url)

CFG = FrontendConfig(
    api_url=_base,
    image_base_dir=args.image_base,
    title_suffix=args.title_suffix,
)


st.set_page_config(page_title=f"MPMQA RAG {CFG.title_suffix}".strip(), layout="wide")
st.title(f"BlueSphere AI Chatbot {CFG.title_suffix}".strip())

# ---------- 会话状态 ----------
if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4().hex[:12]
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- 控制区 ----------
col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
with col1:
    if st.button("清空对话", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.rerun()
with col2:
    mode_label = st.radio(
        "检索模式",
        list(MODE_MAP.keys()),
        index=2,
        label_visibility="collapsed",
    )
with col3:
    stream_mode = st.toggle("流式输出", value=True)
with col4:
    st.caption(f"会话：{st.session_state.session_id} | API：{CFG.api_url}")

mode = MODE_MAP[mode_label]


# ---------- 图片路径解析 ----------
def resolve_image_path(fname):
    """
    解析图片路径：
      1) fname 本身就是可访问的完整相对/绝对路径 → 直接用
      2) 否则当文件名，拼 CFG.image_base_dir 再试
    返回可访问路径或 None。
    """
    if not fname:
        return None
    fname = str(fname).strip()
    if not fname:
        return None
    if os.path.exists(fname):
        return fname
    p = os.path.join(CFG.image_base_dir, fname)
    if os.path.exists(p):
        return p
    return None


def uid_to_path(uid):
    """
    兜底：老命名 {uid}.png 的映射（图名 = chunk uid 时才有效）。
    现在图是 img_xxxx 命名，正常不会命中，保留仅作兼容。
    """
    if uid is None:
        return None
    uid = str(uid)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(CFG.image_base_dir, f"{uid}{ext}")
        if os.path.exists(p):
            return p
    p = os.path.join(CFG.image_base_dir, uid)
    if os.path.exists(p):
        return p
    return None


def build_images_map_from_sources(sources):
    """
    从 sources 构建 uid -> [img_path, ...] 映射。
    兼容字段：images（列表）、image_filename（单个，老字段）。
    """
    m = {}
    for s in sources or []:
        uid = s.get("index")
        imgs = s.get("images") or []
        if not imgs and s.get("image_filename"):
            imgs = [s["image_filename"]]
        if uid and imgs:
            m[str(uid)] = [x for x in imgs if x]
    return m


def render_image(img_path):
    if img_path and os.path.exists(img_path):
        pil_img = Image.open(img_path)
        w, h = pil_img.size
        scale = min(CFG.image_max_w / w, CFG.image_max_h / h)
        new_size = (int(w * scale), int(h * scale))
        image_zoom(
            pil_img,
            mode="dragmove",
            size=new_size,
            zoom_factor=CFG.image_zoom_factor,
            keep_resolution=True,
        )
    else:
        st.warning(f"图片不存在：{img_path}")


def render_sources_only(sources):
    """调试用：展示本轮召回切片，不依赖 uid→图片"""
    with st.expander("查看本轮召回切片（调试）", expanded=False):
        for s in sources:
            st.markdown(
                f"- uid={s.get('index')}  "
                f"cosine_dist={s.get('score')}  "
                f"similarity={s.get('similarity')}  "
                f"bm25={s.get('bm25_score')}  "
                f"rrf={s.get('rrf_score')}"
            )
            st.caption((s.get("document") or "")[:100])




# ---------- 按 [ref:uid] 与 [link:id] 渲染（非流式 / 历史消息） ----------

LINK_PATTERN = re.compile(r"\[link:([\w\-]+)\]")

def render_links(text: str, links_map: dict) -> str:
    """把 [link:id] 替换成 <a>；id 不在 links_map 里则删掉（防幻觉）"""
    def _repl(m):
        link = links_map.get(m.group(1))
        if not link:
            return ""
        return f'<a href="{link["url"]}" target="_blank">{link["name"]}</a>'
    return LINK_PATTERN.sub(_repl, text)

REF_PATTERN = re.compile(r"\[ref:([\w\-]+)\]")


def render_answer_with_refs(answer, sources=None, links=None, seen_uids=None):
    """
    按 [ref:uid] 切分 answer，文字用 st.write，图片按 sources 里的 images 列表逐张渲染。
    seen_uids: 用于跨块/跨轮去重（可选）。传 None 则不去重。
    """
    if seen_uids is None:
        seen_uids = set()

    links_map = {l["id"]: l for l in (links or [])}
    images_map = build_images_map_from_sources(sources)

    parts = REF_PATTERN.split(answer)
    # split 后：文字, uid, 文字, uid, ...
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # 文字部分
            if part.strip():
                st.markdown(render_links(part.strip(), links_map), unsafe_allow_html=True)
        else:
            # uid 部分
            uid = part
            if uid in seen_uids:
                continue
            seen_uids.add(uid)

            fnames = images_map.get(str(uid), [])
            if fnames:
                for fname in fnames:
                    p = resolve_image_path(fname)
                    if p:
                        render_image(p)
                    else:
                        st.warning(f"图片不存在：{fname}")
            else:
                # 兜底：老命名 {uid}.png
                img_path = uid_to_path(uid)
                if img_path:
                    render_image(img_path)


# ---------- 流式渲染核心 ----------
def _safe_prefix_len(buffer):
    """
    返回 buffer 中可以安全输出的前缀长度（不会被后续 token 补全成 ref）。
    策略：若末尾是 '[' / '[r' / '[re' / '[ref' / '[ref:' / '[ref:xx' 这类前缀，截断到 '[' 之前。
    """
    idx = buffer.rfind("[")
    if idx == -1:
        return len(buffer)
    tail = buffer[idx:]
    if re.fullmatch(r"\[(r(e(f(:([\w\-]*)?)?)?)?)?", tail):
        return idx
    return len(buffer)


def _consume_buffer(buffer, seen_uids, render_queue):
    """
    从 buffer 中切出完整 [ref:uid]，把要渲染的内容放入 render_queue。
    render_queue: list of ("text", str) / ("image", uid)
    返回 remaining_buffer
    """
    while True:
        m = REF_PATTERN.search(buffer)
        if not m:
            safe_len = _safe_prefix_len(buffer)
            if safe_len > 0:
                render_queue.append(("text", buffer[:safe_len]))
            buffer = buffer[safe_len:]
            break

        before = buffer[:m.start()]
        if before:
            render_queue.append(("text", before))
        uid = m.group(1)
        if uid not in seen_uids:
            seen_uids.add(uid)
            render_queue.append(("image", uid))
        buffer = buffer[m.end():]

    return buffer


def stream_and_render(question, session_id, mode, user_id):
    """
    流式请求后端，边收 token 边渲染。
    返回 (answer_with_refs, sources)
    """
    answer_with_refs = ""
    seen_uids = set()
    images_map = {}
    links_map = {}
    placeholder = st.empty()
    rendered_text = ""
    text_buffer = ""
    sources = []
    links_map = {}

    try:
        with requests.post(
            STREAM_URL,
            json={"question": question, "session_id": session_id, "mode": mode, "user_id": user_id},
            stream=True,
            timeout=CFG.request_timeout,
            proxies=PROXIES,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()

            event_name = None
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()

                if line == "":
                    event_name = None
                    continue

                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event_name == "start":
                    pass

                elif event_name == "sources":
                    # 【关键】uid -> [img_path, ...]
                    images_map = payload.get("images_map", {}) or {}
                    continue

                elif event_name == "token":
                    text = payload.get("text", "")
                    if not text:
                        continue
                    answer_with_refs += text
                    text_buffer += text

                    render_queue = []
                    text_buffer = _consume_buffer(text_buffer, seen_uids, render_queue)

                    for kind, val in render_queue:
                        if kind == "text":
                            rendered_text += val
                            placeholder.markdown(rendered_text)
                        else:  # image
                            fnames = images_map.get(str(val), [])
                            if fnames:
                                for fname in fnames:
                                    p = resolve_image_path(fname)
                                    if p:
                                        st.image(p, use_container_width=False)
                                    else:
                                        st.warning(f"图片不存在：{fname}")
                            else:
                                # 兜底：老命名
                                img_path = uid_to_path(val)
                                if img_path:
                                    st.image(img_path, use_container_width=False)

                elif event_name == "error":
                    st.error(f"服务端错误：{payload.get('message', '未知错误')}")
                    return answer_with_refs, sources

                elif event_name == "done":
                    links = payload.get("links", []) or []
                    links_map = {l["id"]: l for l in links}
                    if text_buffer.strip():
                        rendered_text += text_buffer
                        placeholder.markdown(rendered_text)
                        text_buffer = ""
                    # 用最终文字 + links 重渲染一遍（覆盖流式中的原始 [link:xx] 文本）
                    placeholder.markdown(
                        render_links(rendered_text, links_map),
                        unsafe_allow_html=True,
                    )  
                    break

    except Exception as e:
        st.error(f"请求失败：{e}")
        return answer_with_refs, sources

    # 流结束后拉一次 sources（用于调试 expander 和历史渲染）
    try:
        r = requests.get(
            SOURCES_URL,
            params={"session_id": session_id},
            timeout=CFG.request_timeout,
            proxies=PROXIES,
        )
        if r.ok:
            sources = r.json().get("sources", [])
    except Exception:
        pass

    return answer_with_refs, sources, links


# ---------- 非流式请求 ----------
def request_non_stream(question, session_id, mode, user_id):
    """
    非流式：一次性拿完整 answer + sources。
    返回 (answer, sources)
    """
    try:
        resp = requests.post(
            ASK_URL,
            json={"question": question, "session_id": session_id, "mode": mode, "user_id": user_id},
            timeout=CFG.request_timeout,
            proxies=PROXIES,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("answer", ""),
            data.get("sources", []),
            data.get("links", []),
        )
    except Exception as e:
        st.error(f"请求失败：{e}")
        return "", []


# ---------- 历史渲染 ----------
for turn in st.session_state.messages:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_answer_with_refs(
            turn["answer"],
            sources=turn.get("sources", []),
            links=turn.get("links", []),
            seen_uids=set(),
        )
        render_sources_only(turn.get("sources", []))


# ---------- 输入 ----------
question = st.chat_input("请输入你的问题")

if question and question.strip():
    question = question.strip()
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if stream_mode:
            answer, sources, links = stream_and_render(
                question=question,
                session_id=st.session_state.session_id,
                mode=mode,
                user_id=st.session_state.user_id,
            )
        else:
            with st.spinner("正在检索并生成回答..."):
                answer, sources, links = request_non_stream(
                    question=question,
                    session_id=st.session_state.session_id,
                    mode=mode,
                    user_id=st.session_state.user_id,
                )
            render_answer_with_refs(answer, sources=sources, links=links, seen_uids=set())
            render_sources_only(sources)

    st.session_state.messages.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "links": links,
    })