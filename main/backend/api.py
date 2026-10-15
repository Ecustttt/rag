import os
import re
import time
import threading
import argparse
from collections import defaultdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document

from config import AppConfig
from retriever import build_embedding, build_vectorstore, Retriever
from chains import (
    build_llm, build_chains, format_history, format_docs,
    dedup_refs, normalize_ref_position, strip_refs,
)


# ======================
# 会话管理
# ======================
class SessionManager:
    def __init__(self, max_history=3, ttl=1800, clean_interval=60):
        self._sessions = defaultdict(list)
        self._last_active = {}
        self.max_history = max_history
        self.ttl = ttl
        self._clean_interval = clean_interval
        self._start_cleaner()

    def add_message(self, session_id, role, content):
        self._sessions[session_id].append({"role": role, "content": content})
        self._last_active[session_id] = time.time()
        self._sessions[session_id] = self._sessions[session_id][-self.max_history:]

    def get_history(self, session_id):
        self._last_active[session_id] = time.time()
        return self._sessions.get(session_id, [])

    def _start_cleaner(self):
        def _clean():
            while True:
                now = time.time()
                expired = [sid for sid, t in self._last_active.items() if now - t > self.ttl]
                for sid in expired:
                    self._sessions.pop(sid, None)
                    self._last_active.pop(sid, None)
                time.sleep(self._clean_interval)
        threading.Thread(target=_clean, daemon=True).start()


# ======================
# 全局状态（延迟初始化）
# ======================
app = FastAPI(title="MPMQA RAG API")

cfg: AppConfig = None
retriever: Retriever = None
llm = None
chains: dict = None
session_mgr: SessionManager = None


def init_app(config: AppConfig):
    """在 __main__ 里调用，完成所有重资源初始化"""
    global cfg, retriever, llm, chains, session_mgr
    cfg = config

    embedding = build_embedding(cfg)
    vectorstore = build_vectorstore(cfg, embedding)
    retriever = Retriever(cfg, embedding, vectorstore)
    llm = build_llm(cfg)
    chains = build_chains(cfg, llm)
    session_mgr = SessionManager(
        max_history=cfg.session_max_history,
        ttl=cfg.session_ttl,
        clean_interval=cfg.session_clean_interval,
    )
    print(f">>> [{cfg.dataset}] 初始化完成, collection={cfg.collection_name}")


# ======================
# API models
# ======================
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    mode: Literal["vector", "bm25", "hybrid"] = "vector"


class SourceItem(BaseModel):
    index: str
    score: float | None
    similarity: float | None
    bm25_score: float | None = None
    rrf_score: float | None = None
    document: str
    metadata: dict
    image_filename: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    images: list[str]
    session_id: str


# ======================
# 路由
# ======================
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id or cfg.default_session_id
    session_mgr.add_message(session_id, "user", query)
    history = session_mgr.get_history(session_id)
    history_msgs = format_history(history)

    intent = (await chains["intent_chain"].ainvoke({
        "question": query, "history": history_msgs,
    })).strip().lower()

    if intent in ("thanks", "chitchat"):
        direct_reply = (await llm.ainvoke([
            ("system", chains["direct_reply_prompt"]),
            *history_msgs,
            ("human", query),
        ])).content.strip()
        session_mgr.add_message(session_id, "assistant", direct_reply)
        return QueryResponse(answer=direct_reply, sources=[], images=[], session_id=session_id)

    search_query = (await chains["rewrite_chain"].ainvoke({
        "question": query, "history": history_msgs,
    })).strip()
    search_query = search_query.strip('"').strip("'")
    search_query = re.sub(r"^(rewritten query|output|search query)\s*[:：]\s*", "", search_query, flags=re.I)
    print(f">>> 原始 query: {query}")
    print(f">>> 改写 query: {search_query}")

    results = await retriever.retrieve(search_query, mode=request.mode, k=cfg.top_k)
    docs_with_score = [
        (Document(page_content=r["text"], metadata=r["meta"]), r) for r in results
    ]
    print(">>> 输入上下文:\n", format_docs(docs_with_score))

    answer = await chains["rag_chain"].ainvoke({
        "question": query,
        "history": history,
        "docs_with_score": docs_with_score,
    })
    answer = normalize_ref_position(answer)
    answer = dedup_refs(answer, docs_with_score)

    clean_answer = strip_refs(answer)
    session_mgr.add_message(session_id, "assistant", clean_answer)

    sources = []
    for d, r in docs_with_score:
        meta = d.metadata
        images_str = meta.get("images", "")
        image_list = images_str.split(",") if images_str else []
        vec_score = r.get("vec_score")
        similarity = round(1.0 - vec_score / 2.0, 4) if vec_score is not None else None
        sources.append(SourceItem(
            index=meta.get("uid"),
            score=vec_score,
            similarity=similarity,
            bm25_score=r.get("bm25_score"),
            rrf_score=r.get("rrf_score"),
            document=d.page_content,
            metadata=meta,
            image_filename=image_list[0] if image_list else None,
        ))

    return QueryResponse(answer=answer, sources=sources, images=[], session_id=session_id)


# ======================
# 入口
# ======================
if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GWM")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()

    init_app(AppConfig(dataset=args.dataset, host=args.host, port=args.port))
    uvicorn.run(app, host=args.host, port=args.port)