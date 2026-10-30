import json
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from tqdm import tqdm

from embedding_client import EmbeddingClient
import argparse


# ======================
# 配置
# ======================
@dataclass
class IndexConfig:
    dataset: str = "GWM"
    chroma_root: str = "./data/chroma_db"
    
    embed_endpoint: str = "http://localhost:8001/v1/embeddings"
    embed_model: str = "qwen3-embedding-0.6b"
    embed_api_key: str = field(default_factory=lambda: os.getenv("EMBED_API_KEY", ""))

    embed_timeout: float = 60.0
    embed_max_retries: int = 3

    batch_size: int = 32
    hnsw_space: str = "cosine"

    @property
    def chroma_path(self): return f"{self.chroma_root}/{self.dataset.lower()}"
    @property
    def collection_name(self): return f"{self.dataset}_docs"
    @property
    def image_prefix_strip(self): return f"./data/images/{self.dataset}/"


# ======================
# 构造入库数据
# ======================
def chunks_from_json(json_path: Path, file_idx: int, cfg: IndexConfig) -> list[dict]:
    doc_name = json_path.stem

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for item in data:
        content = item.get("content", "").strip()
        if not content:
            continue

        metadata = {
            "doc_name": doc_name,
            "uid": f"{file_idx}_{item.get('id')}",
            "chunk_index": item.get("id"),
            "seg_id": item.get("seg_id"),
            "title_path": " > ".join(item.get("title_path", [])),
            "source_file": json_path.name,
        }

        images = item.get("images", [])
        if images:
            cleaned = []
            for img in images:
                img = img.replace("\\", "/")
                if img.startswith(cfg.image_prefix_strip):
                    img = img[len(cfg.image_prefix_strip):]
                cleaned.append(img)
            metadata["images"] = ",".join(cleaned)

        chunks.append({"text": content, "metadata": metadata})
    return chunks


# ======================
# 入库
# ======================
def build_collection(chunks, collection, embed_client: EmbeddingClient, batch_size: int):
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = embed_client.embed(texts)
        ids = [f"{c['metadata']['doc_name']}_{c['metadata']['chunk_index']}" for c in batch]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )


# ======================
# 主流程
# ======================
def run_index(docs_dir: str, cfg: IndexConfig = None):
    cfg = cfg or IndexConfig()

    client = chromadb.PersistentClient(path=cfg.chroma_path)
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": cfg.hnsw_space},
    )

    embed_client = EmbeddingClient(
    cfg.embed_endpoint, cfg.embed_model, cfg.embed_api_key, cfg.embed_timeout, cfg.embed_max_retries)

    json_files = sorted(Path(docs_dir).glob("*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件\n")

    for file_idx, json_path in enumerate(json_files, start=1):
        print(f"处理: {json_path.name}")

        chunks = chunks_from_json(json_path, f"{file_idx:02d}", cfg)
        print(f"  有效切片: {len(chunks)}")

        if chunks:
            build_collection(chunks, collection, embed_client, cfg.batch_size)
            print(f"  ✅ 完成")
        else:
            print(f"  ⚠️ 无有效数据，跳过")

    print(f"\nCollection '{cfg.collection_name}' 共 {collection.count()} 条")
    print("全部完成")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GWM")
    args = ap.parse_args()

    cfg = IndexConfig(dataset=args.dataset)
    run_index(f"./data/docs/{cfg.dataset}_json", cfg)