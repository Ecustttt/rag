import requests


class EmbeddingClient:
    def __init__(self, base_url: str, model: str, timeout: int = 300):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        resp = requests.post(
            self.endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},   # ← 新增
            timeout=self.timeout,
        )

        if not resp.ok:
            # ===== 调试信息 =====
            print("=== EMBED 失败 ===")
            print("status:", resp.status_code)
            print("resp.text:", resp.text[:2000])          # 服务端真实报错
            print(f"batch size: {len(texts)}")
            print("各条长度:", [len(t) for t in texts])    # 找超长/空串
            for idx, t in enumerate(texts):
                if len(t) == 0 or len(t) > 2000:           # 可疑项
                    print(f"  [可疑 {idx}] len={len(t)} head={t[:100]!r} tail={t[-50:]!r}")
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]