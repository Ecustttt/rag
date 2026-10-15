import argparse
import uuid
import re
import os

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

CFG = FrontendConfig(
    api_url=args.api_url,
    image_base_dir=args.image_base,
    title_suffix=args.title_suffix,
)


st.set_page_config(page_title=f"MPMQA RAG {CFG.title_suffix}".strip(), layout="wide")
st.title(f"MPMQA 多轮问答助手 {CFG.title_suffix}".strip())

# ---------- 会话状态 ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- 控制区 ----------
col1, col2, col3 = st.columns([1, 1, 4])
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
    st.caption(f"会话：{st.session_state.session_id} | API：{CFG.api_url}")

mode = MODE_MAP[mode_label]


# ---------- 工具函数 ----------
def render_image(img_path):
    if os.path.exists(img_path):
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


def build_image_map(sources):
    mapping = {}
    for s in sources:
        idx = s.get("index")
        img_name = s.get("image_filename")
        if idx is not None and img_name:
            mapping[idx] = os.path.join(CFG.image_base_dir, img_name)
    return mapping


def render_answer_with_refs(answer, sources):
    image_map = build_image_map(sources)
    parts = re.split(r"(\[ref:[\w\-]+\])", answer)
    for part in parts:
        m = re.match(r"\[ref:([\w\-]+)\]", part)
        if m:
            img_path = image_map.get(m.group(1))
            if img_path:
                render_image(img_path)
        else:
            if part.strip():
                st.write(part.strip())


def render_sources_only(turn):
    with st.expander("查看本轮召回切片（调试）", expanded=False):
        for s in turn["sources"]:
            st.markdown(
                f"- uid={s['index']}  "
                f"cosine_dist={s.get('score')}  "
                f"similarity={s.get('similarity')}  "
                f"bm25={s.get('bm25_score')}  "
                f"rrf={s.get('rrf_score')}"
            )
            st.caption(s.get("document", "")[:100])


# ---------- 历史渲染 ----------
for turn in st.session_state.messages:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_answer_with_refs(turn["answer"], turn["sources"])
        render_sources_only(turn)

# ---------- 输入 ----------
question = st.chat_input("请输入你的问题")

if question and question.strip():
    question = question.strip()
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                resp = requests.post(
                    CFG.api_url,
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "mode": mode,
                    },
                    timeout=CFG.request_timeout,
                    proxies=PROXIES,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"请求失败：{e}")
                st.stop()

        answer = data.get("answer", "")
        sources = data.get("sources", [])
        render_answer_with_refs(answer, sources)
        render_sources_only({"sources": sources})

    st.session_state.messages.append({
        "question": question,
        "answer": answer,
        "sources": sources,
    })