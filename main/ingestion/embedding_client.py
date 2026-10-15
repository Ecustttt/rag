import requests


class EmbeddingClient:
    def __init__(self, base_url: str, model: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        resp = requests.post(
            f"{self.base_url}/v1/embeddings", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]