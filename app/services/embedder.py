import json
from urllib import request, error


class BiEncoderEmbedder:
    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b-fp16",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_embed(self, input_data: str | list[str]) -> dict:
        url = f"{self.base_url}/api/embed"

        payload = {
            "model": self.model,
            "input": input_data,
        }

        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"embedding request failed: {e.code} {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"cannot connect to ollama: {e}") from e

        data = json.loads(raw)
        return data

    def embed_text(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            return []

        embeddings = self.embed_texts([text])
        if not embeddings:
            return []

        return embeddings[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = []
        for text in texts:
            text = text.strip()
            if text:
                clean_texts.append(text)

        if not clean_texts:
            return []

        data = self._post_embed(clean_texts)

        if "embeddings" in data:
            embeddings = data["embeddings"]
        elif "embedding" in data:
            embeddings = [data["embedding"]]
        else:
            raise RuntimeError(f"unexpected embedding response: {data}")

        if len(embeddings) != len(clean_texts):
            raise RuntimeError("embedding count does not match input count")

        return embeddings