"""Tests for the lightweight hybrid reranker."""

from app.schemas.rag import RetrievedChunk
from app.services.reranker import Reranker


def item(text: str, score: float, chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        source="test.md",
        doc_id="test.md",
        chunk_id=chunk_id,
        text=text,
        score=score,
    )


def test_lexical_overlap_can_break_close_vector_scores():
    chunks = [
        item("unrelated material", 0.81, 0),
        item("FastAPI supports dependency injection", 0.80, 1),
    ]
    result = Reranker().rerank("FastAPI dependency injection", chunks, top_n=1)
    assert result[0].chunk_id == 1


if __name__ == "__main__":
    test_lexical_overlap_can_break_close_vector_scores()
    print("RERANKER_TEST=PASS")
