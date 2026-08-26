#保存 chunk 的 embedding 结果，让没有变化的 chunk 不需要重复向量化

import hashlib
import json
from pathlib import Path
from typing import Protocol

from app.schemas.rag import Chunk


class BatchEmbedder(Protocol):
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def _chunk_key(chunk: Chunk, model: str) -> str:
    raw = f"{model}\0{chunk.doc_id}\0{chunk.chunk_id}\0{chunk.text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()#计算chunk的哈希码，确保chunk没有改变


def _load_cache(path: Path, model: str) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))#读取旧的哈希值，如果模型变化或者哈希值为空，就返回空
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("model") != model:
        return {}
    vectors = data.get("vectors", {})
    if not isinstance(vectors, dict):
        return {}
    return vectors


def build_or_reuse_embeddings(
    chunks: list[Chunk],
    embedder: BatchEmbedder,
    cache_path: str | Path,
) -> list[list[float]]:
    path = Path(cache_path)
    model = embedder.model
    old_vectors = _load_cache(path, model)

    keys = [_chunk_key(chunk, model) for chunk in chunks]
    current_vectors: dict[str, list[float]] = {}
    missing_keys: list[str] = []
    missing_texts: list[str] = []
    # 遍历 chunk，优先复用旧缓存；缺失的记录下来，稍后统一补算
    for key, chunk in zip(keys, chunks):
        vector = old_vectors.get(key)
        if isinstance(vector, list) and vector:
            current_vectors[key] = vector
        else:
            missing_keys.append(key)
            missing_texts.append(chunk.text)

    if missing_texts:
        generated = embedder.embed_texts(missing_texts)
        if len(generated) != len(missing_texts):#如果数量对不上，说明 embedding 服务返回异常，不能继续，否则文本和向量会错位
            raise RuntimeError("embedding count does not match cache misses")
        for key, vector in zip(missing_keys, generated):
            if not vector:
                raise RuntimeError("embedding service returned an empty vector")
            current_vectors[key] = vector
    #如果 app/index 目录不存在，就创建
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"model": model, "vectors": current_vectors},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return [current_vectors[key] for key in keys]#按照原本的顺序返回
