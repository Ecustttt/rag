import re
import os
from typing import Literal

from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import AppConfig


def build_embedding(cfg: AppConfig) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=cfg.embed_model,
        openai_api_base=cfg.vllm_embed_url,
        openai_api_key="not-needed",
        check_embedding_ctx_length=False,
    )


def build_vectorstore(cfg: AppConfig, embedding) -> Chroma:
    return Chroma(
        persist_directory=cfg.chroma_path,
        embedding_function=embedding,
        collection_name=cfg.collection_name,
    )


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, cfg: AppConfig, embedding, vectorstore):
        self.cfg = cfg
        self.embedding = embedding
        self.vectorstore = vectorstore

        data = vectorstore.get(include=["documents", "metadatas"])
        self.bm25_ids = data["ids"]
        self.bm25_docs = {
            i: {"text": d, "meta": m}
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
        }
        self.bm25 = BM25Okapi([tokenize(d) for d in data["documents"]])
        print(f">>> BM25 索引已建，共 {len(self.bm25_ids)} 条")

    async def vector_search(self, query, k):
        res = self.vectorstore._collection.query(
            query_embeddings=[self.embedding.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            if dist <= self.cfg.score_threshold:
                out.append({
                    "id": cid, "text": doc, "meta": meta,
                    "vec_score": float(dist), "bm25_score": None, "rrf_score": None,
                })
        return out

    def bm25_search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            cid = self.bm25_ids[i]
            out.append({
                "id": cid,
                "text": self.bm25_docs[cid]["text"],
                "meta": self.bm25_docs[cid]["meta"],
                "vec_score": None, "bm25_score": float(scores[i]), "rrf_score": None,
            })
        return out

    def rrf_fuse(self, vec_results, bm25_results, k):
        scores, payload = {}, {}
        for rank, item in enumerate(vec_results, start=1):
            cid = item["id"]
            scores[cid] = scores.get(cid, 0) + 1 / (self.cfg.rrf_k + rank)
            payload[cid] = dict(item)
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["id"]
            scores[cid] = scores.get(cid, 0) + 1 / (self.cfg.rrf_k + rank)
            if cid in payload:
                payload[cid]["bm25_score"] = item["bm25_score"]
            else:
                payload[cid] = dict(item)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        for cid, s in ranked:
            payload[cid]["rrf_score"] = s
        return [payload[cid] for cid, _ in ranked]

    async def retrieve(
        self,
        semantic_query: str | None = None,
        keyword_query: str | None = None,
        mode: Literal["vector", "bm25", "hybrid"] = "vector",
        k: int | None = None,
        query: str | None = None,
    ):
        # 向后兼容旧调用
        if semantic_query is None:
            semantic_query = query if query is not None else (keyword_query or "")
        if keyword_query is None:
            keyword_query = query if query is not None else semantic_query

        k = k or self.cfg.top_k

        if mode == "vector":
            return await self.vector_search(semantic_query, k)
        if mode == "bm25":
            return self.bm25_search(keyword_query, k)
        vec = await self.vector_search(semantic_query, k)
        bm  = self.bm25_search(keyword_query, k)
        return self.rrf_fuse(vec, bm, k)