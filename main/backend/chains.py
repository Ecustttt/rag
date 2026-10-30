import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import AppConfig


REF_PATTERN = r"\[ref:([\w\-]+)\]"


def build_llm(cfg: AppConfig) -> ChatOpenAI:
    extra = {}
    if cfg.supports_chat_template_kwargs:          # 只有 vLLM 才带
        extra["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}
        }
    return ChatOpenAI(
        model=cfg.chat_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        openai_api_base=cfg.chat_base_url,
        openai_api_key=cfg.chat_api_key or "not-needed",
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries, 
        stream_usage=True,
        **extra,
    )


def format_history(history: list) -> list:
    msgs = []
    for msg in history:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            msgs.append(AIMessage(content=msg["content"]))
    return msgs


LINK_PATTERN = r"\[link:([\w\-]+)\]"

def format_links(links):
    if not links:
        return "（无可用链接）"
    return "\n".join(f"[{l['id']}] {l['name']}" for l in links)

def extract_used_links(text, links_map):
    """从 answer 提取实际用到的 id，返回去重后的链接对象列表；不在表里的 id 忽略"""
    used, seen = [], set()
    for m in re.finditer(LINK_PATTERN, text):
        lid = m.group(1)
        if lid in links_map and lid not in seen:
            seen.add(lid)
            used.append({
                "id": links_map[lid]["id"],
                "name": links_map[lid]["name"],
                "url": links_map[lid]["url"],
            })
    return used

def strip_link_refs(text, links_map):
    """删除不在 links_map 中的 [link:xxx]（防幻觉），合法标记保留"""
    return re.sub(LINK_PATTERN, lambda m: m.group(0) if m.group(1) in links_map else "", text)


def format_docs(docs_with_score):
    texts = []
    for item in docs_with_score:
        d = item[0] if isinstance(item, tuple) else item
        idx = d.metadata.get("uid", "unknown")
        texts.append(f"[source_id: {idx}] {d.page_content}")
    return "\n\n".join(texts)


def build_chains(cfg: AppConfig, llm):
    system_prompt = cfg.load_prompt(cfg.system_prompt_file)
    rewrite_prompt = cfg.load_prompt(cfg.rewrite_prompt_file)
    intent_prompt = cfg.load_prompt(cfg.intent_prompt_file)
    direct_reply_prompt = cfg.load_prompt(cfg.direct_reply_prompt_file)
    rewrite_vector_prompt = cfg.load_prompt(cfg.rewrite_vector_prompt_file)
    rewrite_bm25_prompt = cfg.load_prompt(cfg.rewrite_bm25_prompt_file)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "chunks：{context}\n\nlinks：{link_list}\n\nquestion：{question}"),
    ])
    rag_chain = (
        {
            "context": lambda x: format_docs(x["docs_with_score"]),
            "question": lambda x: x["question"],
            "history": lambda x: format_history(x["history"]),
            "link_list": lambda x: format_links(x["links"]),
        }
        | rag_prompt | llm
    )

    intent_chain = (
        ChatPromptTemplate.from_messages([
            ("system", intent_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm
    )

    rewrite_vector_chain = (
        ChatPromptTemplate.from_messages([
            ("system", rewrite_vector_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm
    )
    rewrite_bm25_chain = (
        ChatPromptTemplate.from_messages([
            ("system", rewrite_bm25_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm
    )

    rewrite_chain = (
        ChatPromptTemplate.from_messages([
            ("system", rewrite_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm
    )

    return {
        "rag_chain": rag_chain,
        "intent_chain": intent_chain,
        "rewrite_chain": rewrite_chain,
        "rewrite_vector_chain": rewrite_vector_chain,
        "rewrite_bm25_chain": rewrite_bm25_chain,
        "direct_reply_prompt": direct_reply_prompt,
    }


def parse_output(text: str):
    if not text:
        return "", []
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return raw.strip(), []
    answer = str(data.get("answer", "")).strip()
    sources_raw = data.get("sources", [])
    source_chunks = [str(x).strip() for x in sources_raw if str(x).strip()] if isinstance(sources_raw, list) else []
    return answer, source_chunks


def dedup_refs(text, docs_with_score):
    has_image = set()
    for d, _ in docs_with_score:
        idx = str(d.metadata.get("uid", ""))
        if idx and d.metadata.get("images", ""):
            has_image.add(idx)
    seen = set()
    def replacer(m):
        ref_id = m.group(1)
        if ref_id not in has_image or ref_id in seen:
            return ""
        seen.add(ref_id)
        return m.group(0)
    return re.sub(REF_PATTERN, replacer, text)


def normalize_ref_position(text):
    """连续 ref 串后的标点挪到串前：[ref:a][ref:b]. → .[ref:a][ref:b]"""
    return re.sub(
        r"((?:\[ref:[\w\-]+\])+)([.。!！?？,，;；])",
        r"\2\1",
        text,
    )


def strip_refs(text):
    return re.sub(REF_PATTERN, "", text).strip()


def clean_query(text: str, fallback: str) -> str:
    """清理改写输出：去代码围栏、引号、前缀标签；空则回退原 query"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    t = t.strip('"').strip("'").strip()
    t = re.sub(
        r"^(rewritten query|output|search query|semantic|keyword)\s*[:：]\s*",
        "", t, flags=re.I,
    ).strip()
    return t or fallback