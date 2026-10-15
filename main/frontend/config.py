from dataclasses import dataclass


@dataclass
class FrontendConfig:
    api_url: str = "http://127.0.0.1:8002/ask"
    image_base_dir: str = "./data/images/GWM/"
    title_suffix: str = ""

    request_timeout: int = 600
    image_max_w: int = 600
    image_max_h: int = 800
    image_zoom_factor: float = 2.0


MODE_MAP = {"语义检索": "vector", "精确检索": "bm25", "混合检索": "hybrid"}
PROXIES = {"http": None, "https": None}