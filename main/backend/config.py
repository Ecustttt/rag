from pathlib import Path
from dataclasses import dataclass
import os


@dataclass
class AppConfig:
    dataset: str = "GWM"
    base_dir: Path = Path(__file__).resolve().parent

    # 路径（property 动态生成）
    @property
    def image_base_dir(self): return f"./data/images/{self.dataset}/"
    @property
    def chroma_path(self): return f"./data/chroma_db/{self.dataset.lower()}"
    @property
    def collection_name(self): return f"{self.dataset}_docs"

    # 服务地址
    vllm_embed_url: str = "http://localhost:8001/v1"
    vllm_chat_url: str = "http://localhost:8000/v1"

    # 模型
    embed_model: str = "qwen3-embedding-0.6b"
    chat_model: str = "qwen3.5-4b"

    # 检索
    top_k: int = 10
    score_threshold: float = 1.0
    rrf_k: int = 60

    # LLM
    temperature: float = 0.7
    max_tokens: int = 1024
    enable_thinking: bool = False

    # prompt 文件
    prompt_dir: str = "prompts"
    system_prompt_file: str = "system_prompt_en.txt"
    rewrite_prompt_file: str = "rewrite_prompt_en.txt"
    intent_prompt_file: str = "intent_prompt_en.txt"
    direct_reply_prompt_file: str = "direct_reply_prompt_en.txt"
    rewrite_vector_prompt_file: str = "rewrite_prompt_en_vector.txt"
    rewrite_bm25_prompt_file: str = "rewrite_prompt_en_bm25.txt"

    # 会话
    session_max_history: int = 3
    session_ttl: int = 1800
    session_clean_interval: int = 60
    default_session_id: str = "default"

    # server
    host: str = "0.0.0.0"
    port: int = 8002

    def load_prompt(self, filename):
        return (self.base_dir / self.prompt_dir / filename).read_text(encoding="utf-8").strip()