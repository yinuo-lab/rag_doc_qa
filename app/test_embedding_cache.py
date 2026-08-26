"""Embedding cache tests that do not require Ollama."""

from pathlib import Path
from tempfile import TemporaryDirectory

from app.schemas.rag import Chunk
from app.services.embedding_cache import build_or_reuse_embeddings


class FakeEmbedder:
    model = "fake-embedding-model"

    def __init__(self):
        self.batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


def chunk(doc_id: str, chunk_id: int, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, source=doc_id, chunk_id=chunk_id, text=text)


def test_only_changed_chunks_are_embedded():
    embedder = FakeEmbedder()
    with TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "embeddings.json"
        first = [chunk("a.md", 0, "alpha"), chunk("b.md", 0, "beta")]
        vectors1 = build_or_reuse_embeddings(first, embedder, cache_path)
        assert len(vectors1) == 2
        assert embedder.batches == [["alpha", "beta"]]

        vectors2 = build_or_reuse_embeddings(first, embedder, cache_path)
        assert vectors2 == vectors1
        assert len(embedder.batches) == 1

        changed = [chunk("a.md", 0, "alpha changed")]
        build_or_reuse_embeddings(changed, embedder, cache_path)
        assert embedder.batches[-1] == ["alpha changed"]


if __name__ == "__main__":
    test_only_changed_chunks_are_embedded()
    print("EMBEDDING_CACHE_TEST=PASS")
