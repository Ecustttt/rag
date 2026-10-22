import asyncio
from langchain_core.documents import Document

from chains import (
    format_history, format_docs, dedup_refs,
    normalize_ref_position, strip_refs, clean_query,
)


class RagService:
    def __init__(self, cfg, chains, llm, retriever):
        self.cfg = cfg
        self.chains = chains
        self.llm = llm
        self.retriever = retriever

    async def detect_intent(self, question, history):
        """返回 'rag' / 'thanks' / 'chitchat'"""
        intent = (await self.chains["intent_chain"].ainvoke({
            "question": question,
            "history": format_history(history),
        })).strip().lower()
        return intent if intent in ("thanks", "chitchat") else "rag"

    async def direct_reply(self, question, history):
        """闲聊/感谢的直接回复（非流式）"""
        return (await self.llm.ainvoke([
            ("system", self.chains["direct_reply_prompt"]),
            *format_history(history),
            ("human", question),
        ])).content.strip()

    async def direct_reply_stream(self, question, history):
        """闲聊/感谢的直接回复（流式）"""
        async for chunk in self.llm.astream([
            ("system", self.chains["direct_reply_prompt"]),
            *format_history(history),
            ("human", question),
        ]):
            yield chunk

    async def rewrite(self, question, history):
        """双路改写，返回 (sem_q, kw_q)"""
        sem_raw, kw_raw = await asyncio.gather(
            self.chains["rewrite_vector_chain"].ainvoke({
                "question": question, "history": format_history(history),
            }),
            self.chains["rewrite_bm25_chain"].ainvoke({
                "question": question, "history": format_history(history),
            }),
        )
        return clean_query(sem_raw, question), clean_query(kw_raw, question)

    async def retrieve(self, sem_q, kw_q, mode):
        """检索，返回 docs_with_score"""
        results = await self.retriever.retrieve(
            semantic_query=sem_q, keyword_query=kw_q,
            mode=mode, k=self.cfg.top_k,
        )
        return [
            (Document(page_content=r["text"], metadata=r["meta"]), r)
            for r in results
        ]

    async def answer(self, question, history, mode="hybrid"):
        """
        非流式 RAG 流程。
        返回 dict: {"answer": str, "sources": list, "intent": str,
                    "sem_query": str, "kw_query": str}
        """
        intent = await self.detect_intent(question, history)

        if intent in ("thanks", "chitchat"):
            reply = await self.direct_reply(question, history)
            return {
                "answer": reply, "sources": [],
                "intent": intent, "sem_query": None, "kw_query": None,
            }

        sem_q, kw_q = await self.rewrite(question, history)
        docs_with_score = await self.retrieve(sem_q, kw_q, mode)

        raw_answer = await self.chains["rag_chain"].ainvoke({
            "question": question,
            "history": history,
            "docs_with_score": docs_with_score,
        })
        raw_answer = normalize_ref_position(raw_answer)
        raw_answer = dedup_refs(raw_answer, docs_with_score)

        # sources 的构建放在 service 之外（main.py 里），
        # 因为不同调用方可能想要不同格式。这里返回原始 docs_with_score。
        return {
            "answer": raw_answer,
            "docs_with_score": docs_with_score,
            "intent": intent,
            "sem_query": sem_q, "kw_query": kw_q,
        }

    async def answer_stream(self, question, history, mode="hybrid"):
        """
        流式 RAG 流程。yield 结构化事件 dict：
          {"event": "intent",   "intent": "rag"}
          {"event": "queries",  "sem_query": ..., "kw_query": ...}
          {"event": "docs",     "docs_with_score": [...]}   # 给调用方构建 sources
          {"event": "token",    "text": "..."}
          {"event": "done",     "raw_answer": "..."}
        调用方负责把这些转成 SSE / 存 source / 写日志。
        """
        intent = await self.detect_intent(question, history)
        yield {"event": "intent", "intent": intent}

        if intent in ("thanks", "chitchat"):
            async for chunk in self.direct_reply_stream(question, history):
                yield {"event": "raw_chunk", "chunk": chunk}
            yield {"event": "done", "raw_answer": None}
            return

        sem_q, kw_q = await self.rewrite(question, history)
        yield {"event": "queries", "sem_query": sem_q, "kw_query": kw_q}

        docs_with_score = await self.retrieve(sem_q, kw_q, mode)
        yield {"event": "docs", "docs_with_score": docs_with_score}

        raw_answer = ""
        async for chunk in self.chains["rag_chain"].astream({
            "question": question,
            "history": history,
            "docs_with_score": docs_with_score,
        }):
            raw_answer += str(getattr(chunk, "content", chunk) or "")
            yield {"event": "raw_chunk", "chunk": chunk}

        yield {"event": "done", "raw_answer": raw_answer}