import os
import re
import time
import json
import threading
import argparse
import asyncio
from collections import defaultdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.documents import Document

from config import AppConfig
from retriever import build_embedding, build_vectorstore, Retriever
from chains import (
    build_llm, build_chains, format_docs,
    dedup_refs, normalize_ref_position, strip_refs,
    clean_query,
)
from rag_service import RagService
from obs import Trace, log


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


class SourceStore:
    def __init__(self, max_turns=20):
        self._store = defaultdict(list)
        self.max_turns = max_turns

    def add(self, session_id, sources_payload):
        self._store[session_id].append(sources_payload)
        self._store[session_id] = self._store[session_id][-self.max_turns:]

    def get_latest(self, session_id):
        items = self._store.get(session_id, [])
        return items[-1] if items else []

    def clear(self, session_id):
        self._store.pop(session_id, None)


# ======================
# 全局状态
# ======================
app = FastAPI(title="MPMQA RAG API")

cfg: AppConfig = None
retriever: Retriever = None
llm = None
chains: dict = None
session_mgr: SessionManager = None
source_store: SourceStore = None
rag_service: RagService = None


def init_app(config: AppConfig):
    global cfg, retriever, llm, chains, session_mgr, source_store, rag_service
    cfg = config

    embedding = build_embedding(cfg)
    vectorstore = build_vectorstore(cfg, embedding)
    retriever = Retriever(cfg, embedding, vectorstore)
    llm = build_llm(cfg)
    chains = build_chains(cfg, llm)
    rag_service = RagService(cfg, chains, llm, retriever)
    session_mgr = SessionManager(
        max_history=cfg.session_max_history,
        ttl=cfg.session_ttl,
        clean_interval=cfg.session_clean_interval,
    )
    source_store = SourceStore()
    print(f">>> [{cfg.dataset}] 初始化完成, collection={cfg.collection_name}")


# ======================
# API models
# ======================
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    mode: Literal["vector", "bm25", "hybrid"] = "vector"
    user_id: str | None = None


class SourceItem(BaseModel):
    index: str
    score: float | None
    similarity: float | None
    bm25_score: float | None = None
    rrf_score: float | None = None
    document: str
    metadata: dict
    image_filename: str | None = None
    images: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str


# ======================
# 公共函数
# ======================
def build_sources_payload(docs_with_score):
    sources = []
    for d, r in docs_with_score:
        meta = d.metadata
        images_str = meta.get("images", "")
        image_list = images_str.split(",") if images_str else []
        vec_score = r.get("vec_score")
        similarity = round(1.0 - vec_score / 2.0, 4) if vec_score is not None else None
        sources.append({
            "index": meta.get("uid"),
            "score": vec_score,
            "similarity": similarity,
            "bm25_score": r.get("bm25_score"),
            "rrf_score": r.get("rrf_score"),
            "document": d.page_content,
            "metadata": meta,
            "image_filename": image_list[0] if image_list else None,
            "images": image_list,
        })
    return sources


def sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def extract_text(chunk) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "content"):
        c = chunk.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(c)
    try:
        return str(chunk)
    except Exception:
        return ""


def build_images_map(sources_payload) -> dict:
    m = {}
    for s in sources_payload:
        uid = s.get("index")
        imgs = s.get("images") or []
        if uid and imgs:
            m[str(uid)] = imgs
    return m


# ======================
# 路由：非流式
# ======================
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id or cfg.default_session_id
    trace = Trace(session_id, query, request.mode, stream=False, user_id=request.user_id)
    trace.set(model=cfg.chat_model)

    try:
        # 壳层：取历史、存当前 query
        history = session_mgr.get_history(session_id)
        session_mgr.add_message(session_id, "user", query)

        trace.mark("intent_start")
        # 调 service（内部会先做 intent + rewrite + retrieve + generate）
        trace.mark("gen_start")
        result = await rag_service.answer(query, history, request.mode)
        trace.mark("gen_end")

        intent = result["intent"]
        clean_answer = strip_refs(result["answer"])
        session_mgr.add_message(session_id, "assistant", clean_answer)

        docs_with_score = result.get("docs_with_score", [])
        sources_payload = build_sources_payload(docs_with_score)
        source_store.add(session_id, sources_payload)

        trace.set(
            intent=intent,
            sem_query=result.get("sem_query"),
            kw_query=result.get("kw_query"),
            recall_count=len(docs_with_score),
            recall_uids=[d.metadata.get("uid") for d, _ in docs_with_score],
            answer_len=len(clean_answer),
            answer=clean_answer[:2000],
        )
        trace.emit()

        sources = [SourceItem(**s) for s in sources_payload]
        return QueryResponse(
            answer=result["answer"], sources=sources, session_id=session_id,
        )

    except HTTPException:
        trace.emit()
        raise
    except Exception as e:
        trace.emit(error=e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# 路由：流式
# ======================
@app.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id or cfg.default_session_id
    trace = Trace(session_id, query, request.mode, stream=True, user_id=request.user_id)
    trace.set(model=cfg.chat_model)

    # 壳层：取历史、存当前 query
    history = session_mgr.get_history(session_id)
    session_mgr.add_message(session_id, "user", query)

    async def event_generator():
        full_answer = ""
        intent = "rag"
        sem_q = kw_q = None
        sources_payload = []

        try:
            trace.mark("gen_start")
            first_token = False

            async for ev in rag_service.answer_stream(query, history, request.mode):
                kind = ev["event"]

                if kind == "intent":
                    intent = ev["intent"]
                    trace.set(intent=intent)
                    if intent in ("thanks", "chitchat"):
                        source_store.add(session_id, [])
                        yield sse_format("start", {"session_id": session_id, "has_sources": False})
                        yield sse_format("sources", {"session_id": session_id, "images_map": {}})

                elif kind == "queries":
                    sem_q, kw_q = ev["sem_query"], ev["kw_query"]
                    trace.set(sem_query=sem_q, kw_query=kw_q)

                elif kind == "docs":
                    sources_payload = build_sources_payload(ev["docs_with_score"])
                    source_store.add(session_id, sources_payload)
                    trace.set(
                        recall_count=len(sources_payload),
                        recall_uids=[s["index"] for s in sources_payload],
                    )
                    yield sse_format("start", {
                        "session_id": session_id,
                        "has_sources": len(sources_payload) > 0,
                    })
                    yield sse_format("sources", {
                        "session_id": session_id,
                        "images_map": build_images_map(sources_payload),
                    })

                elif kind == "raw_chunk":
                    text = extract_text(ev["chunk"])
                    if text:
                        if not first_token:
                            trace.mark("gen_first_token")
                            first_token = True
                        full_answer += text
                        yield sse_format("token", {"text": text})

                elif kind == "done":
                    trace.mark("gen_end")
                    clean_answer = strip_refs(full_answer)
                    session_mgr.add_message(session_id, "assistant", clean_answer)
                    trace.set(
                        answer_len=len(clean_answer),
                        answer=clean_answer[:2000],
                    )
                    trace.emit()
                    yield sse_format("done", {
                        "session_id": session_id,
                        "answer_length": len(full_answer),
                    })

        except Exception as e:
            trace.emit(error=e)
            yield sse_format("error", {"message": f"生成失败: {e}"})
        finally:
            if not trace._done:
                trace.emit(status="aborted")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ======================
# sources 拉取（现在自测用，将来后端接手可删）
# ======================
@app.get("/sources")
async def get_sources(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    return {"session_id": session_id, "sources": source_store.get_latest(session_id)}



# ======================
# 新增：给后端同事调的纯 RAG 接口
# ======================
class RagRequest(BaseModel):
    question: str
    history: list[dict] = []
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    session_id: str | None = None
    user_id: str | None = None


class RagResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    intent: str
    sem_query: str | None
    kw_query: str | None
    trace_id: str 
    trace: dict

@app.post("/rag/answer", response_model=RagResponse)
async def rag_answer(request: RagRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    trace = Trace(
        request.session_id or "", query, request.mode,
        stream=False, user_id=request.user_id,
    )
    trace.set(model=cfg.chat_model)

    try:
        trace.mark("gen_start")
        result = await rag_service.answer(query, request.history, request.mode)
        trace.mark("gen_end")

        docs_with_score = result.get("docs_with_score", [])
        sources_payload = build_sources_payload(docs_with_score)
        clean_answer = strip_refs(result["answer"])

        trace.set(
            intent=result["intent"],
            sem_query=result.get("sem_query"),
            kw_query=result.get("kw_query"),
            recall_count=len(docs_with_score),
            recall_uids=[d.metadata.get("uid") for d, _ in docs_with_score],
            answer_len=len(clean_answer),
            answer=clean_answer[:2000],
        )

        trace_data = trace.emit()

        return RagResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in sources_payload],
            intent=result["intent"],
            sem_query=result.get("sem_query"),
            kw_query=result.get("kw_query"),
            trace_id=trace.trace_id,
            trace=trace_data,
        )

    except HTTPException:
        trace.emit()
        raise
    except Exception as e:
        trace.emit(error=e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/answer/stream")
async def rag_answer_stream(request: RagRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    trace = Trace(
        request.session_id or "", query, request.mode,
        stream=True, user_id=request.user_id,
    )
    trace.set(model=cfg.chat_model)

    async def event_generator():
        full_answer = ""
        intent = "rag"
        sem_q = kw_q = None
        first_token = False

        try:
            trace.mark("gen_start")
            async for ev in rag_service.answer_stream(query, request.history, request.mode):
                kind = ev["event"]

                if kind == "intent":
                    intent = ev["intent"]
                    trace.set(intent=intent)
                    yield sse_format("start", {
                        "session_id": request.session_id,
                        "has_sources": intent == "rag",
                    })

                elif kind == "queries":
                    sem_q, kw_q = ev["sem_query"], ev["kw_query"]
                    trace.set(sem_query=sem_q, kw_query=kw_q)

                elif kind == "docs":
                    docs_with_score = ev["docs_with_score"]
                    sources_payload = build_sources_payload(docs_with_score)
                    trace.set(
                        recall_count=len(sources_payload),
                        recall_uids=[s["index"] for s in sources_payload],
                    )
                    yield sse_format("sources", {
                        "session_id": request.session_id,
                        "images_map": build_images_map(sources_payload),
                    })

                elif kind == "raw_chunk":
                    text = extract_text(ev["chunk"])
                    if text:
                        if not first_token:
                            trace.mark("gen_first_token")
                            first_token = True
                        full_answer += text
                        yield sse_format("token", {"text": text})

                elif kind == "done":
                    trace.mark("gen_end")
                    clean_answer = strip_refs(full_answer)
                    trace.set(
                        answer_len=len(clean_answer),
                        answer=clean_answer[:2000],
                    )
                    trace_data = trace.emit()
                    yield sse_format("done", {
                        "trace_id": trace.trace_id,
                        "answer_length": len(full_answer),
                        "trace": trace_data,
                    })

        except Exception as e:
            trace.emit(error=e)
            yield sse_format("error", {"message": f"生成失败: {e}"})
        finally:
            if not trace._done:
                trace.emit(status="aborted")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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