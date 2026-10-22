import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from config import AppConfig


REF_PATTERN = r"\[ref:([\w\-]+)\]"


def build_llm(cfg: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.chat_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        openai_api_base=cfg.vllm_chat_url,
        openai_api_key="not-needed",
        extra_body={"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}},
    )


def format_history(history: list) -> list:
    msgs = []
    for msg in history:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            msgs.append(AIMessage(content=msg["content"]))
    return msgs


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
        ("human", "chunks：{context}\n\nquestion：{question}"),
    ])
    rag_chain = (
        {
            "context": lambda x: format_docs(x["docs_with_score"]),
            "question": lambda x: x["question"],
            "history": lambda x: format_history(x["history"]),
        }
        | rag_prompt | llm | StrOutputParser()
    )

    intent_chain = (
        ChatPromptTemplate.from_messages([
            ("system", intent_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm | StrOutputParser()
    )

    rewrite_vector_chain = (
        ChatPromptTemplate.from_messages([
            ("system", rewrite_vector_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm | StrOutputParser()
    )
    rewrite_bm25_chain = (
        ChatPromptTemplate.from_messages([
            ("system", rewrite_bm25_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm | StrOutputParser()
    )

    rewrite_chain = (
        ChatPromptTemplate.from_messages([
            ("system", rewrite_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]) | llm | StrOutputParser()
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