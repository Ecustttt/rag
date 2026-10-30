import asyncio
from langchain_core.documents import Document

from chains import (
    format_history, format_docs, dedup_refs,
    normalize_ref_position, strip_refs, clean_query, strip_link_refs, extract_used_links, format_links
)


class RagService:
    def __init__(self, cfg, chains, llm, retriever, links_data, links_map):
        self.cfg = cfg
        self.chains = chains
        self.llm = llm
        self.retriever = retriever
        self.links_data = links_data or []
        self.links_map = links_map or {}

    async def detect_intent(self, question, history):
        """返回 (intent, usage_metadata)"""
        resp = await self.chains["intent_chain"].ainvoke({
            "question": question,
            "history": format_history(history),
        })
        intent = str(resp.content or "").strip().lower()
        intent = intent if intent in ("thanks", "chitchat") else "rag"
        return intent, resp.usage_metadata

    async def direct_reply(self, question, history):
        """闲聊/感谢的直接回复（非流式），返回 (text, usage_metadata)"""
        resp = await self.llm.ainvoke([
            ("system", self.chains["direct_reply_prompt"]),
            *format_history(history),
            ("human", question),
        ])
        return str(resp.content or "").strip(), resp.usage_metadata

    async def direct_reply_stream(self, question, history):
        """闲聊/感谢的直接回复（流式），yield chunk"""
        async for chunk in self.llm.astream([
            ("system", self.chains["direct_reply_prompt"]),
            *format_history(history),
            ("human", question),
        ]):
            yield chunk

    async def rewrite(self, question, history):
        """双路改写，返回 (sem_q, kw_q, sem_usage, kw_usage)"""
        sem_resp, kw_resp = await asyncio.gather(
            self.chains["rewrite_vector_chain"].ainvoke({
                "question": question, "history": format_history(history),
            }),
            self.chains["rewrite_bm25_chain"].ainvoke({
                "question": question, "history": format_history(history),
            }),
        )
        sem_q = clean_query(str(sem_resp.content or ""), question)
        kw_q = clean_query(str(kw_resp.content or ""), question)
        return sem_q, kw_q, sem_resp.usage_metadata, kw_resp.usage_metadata

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
        intent, intent_usage = await self.detect_intent(question, history)
        usage = {"intent": intent_usage}

        if intent in ("thanks", "chitchat"):
            reply, reply_usage = await self.direct_reply(question, history)
            usage["direct"] = reply_usage
            return {
                "answer": reply, "sources": [], "docs_with_score": [],
                "intent": intent, "sem_query": None, "kw_query": None,
                "usage": usage,
                "links": [],
            }

        sem_q, kw_q, sem_usage, kw_usage = await self.rewrite(question, history)
        usage["rewrite_v"] = sem_usage
        usage["rewrite_b"] = kw_usage

        docs_with_score = await self.retrieve(sem_q, kw_q, mode)

        resp = await self.chains["rag_chain"].ainvoke({
            "question": question,
            "history": history,
            "docs_with_score": docs_with_score,
            "links": self.links_data,
        })
        usage["rag"] = resp.usage_metadata
        raw_answer = str(resp.content or "")

        raw_answer = normalize_ref_position(raw_answer)
        raw_answer = dedup_refs(raw_answer, docs_with_score)
        raw_answer = strip_link_refs(raw_answer, self.links_map)
        used_links = extract_used_links(raw_answer, self.links_map)

        return {
            "answer": raw_answer,
            "docs_with_score": docs_with_score,
            "intent": intent,
            "sem_query": sem_q, "kw_query": kw_q,
            "usage": usage,
            "links": used_links,
        }

    async def answer_stream(self, question, history, mode="hybrid"):
        intent, intent_usage = await self.detect_intent(question, history)
        yield {"event": "intent", "intent": intent, "usage": intent_usage}

        if intent in ("thanks", "chitchat"):
            final_usage = None
            async for chunk in self.direct_reply_stream(question, history):
                if getattr(chunk, "usage_metadata", None):
                    final_usage = chunk.usage_metadata
                yield {"event": "raw_chunk", "chunk": chunk}
            yield {"event": "done", "raw_answer": None,
                   "usage": {"direct": final_usage},
                   "links": []}
            return

        sem_q, kw_q, sem_usage, kw_usage = await self.rewrite(question, history)
        yield {
            "event": "queries",
            "sem_query": sem_q, "kw_query": kw_q,
            "usage": {"rewrite_v": sem_usage, "rewrite_b": kw_usage},
        }

        docs_with_score = await self.retrieve(sem_q, kw_q, mode)
        yield {"event": "docs", "docs_with_score": docs_with_score}

        raw_answer = ""
        rag_usage = None
        async for chunk in self.chains["rag_chain"].astream({
            "question": question,
            "history": history,
            "docs_with_score": docs_with_score,
            "links": self.links_data,
        }):
            if getattr(chunk, "usage_metadata", None):
                rag_usage = chunk.usage_metadata
            raw_answer += str(getattr(chunk, "content", chunk) or "")
            yield {"event": "raw_chunk", "chunk": chunk}

        used_links = extract_used_links(raw_answer, self.links_map)
        yield {"event": "done", "raw_answer": raw_answer,
               "usage": {"rag": rag_usage},
               "links": used_links}