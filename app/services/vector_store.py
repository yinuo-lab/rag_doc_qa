from math import sqrt

from app.schemas.rag import Chunk, RetrievedChunk


class InMemoryVectorStore:
    def __init__(self):
        self.records: list[tuple[Chunk, list[float]]] = []

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings size mismatch")

        for chunk, embedding in zip(chunks, embeddings):
            if not chunk.text.strip():
                continue
            if not embedding:
                continue
            self.records.append((chunk, embedding))

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_norm = 0.0
        for x in query_embedding:
            query_norm += x * x
        query_norm = sqrt(query_norm)

        if query_norm == 0:
            return []

        retrieved_chunks: list[RetrievedChunk] = []

        for chunk, embedding in self.records:
            if len(embedding) != len(query_embedding):
                raise ValueError("embedding dimension mismatch")

            emb_norm = 0.0
            for x in embedding:
                emb_norm += x * x
            emb_norm = sqrt(emb_norm)

            if emb_norm == 0:
                continue

            dot = 0.0
            for q, e in zip(query_embedding, embedding):
                dot += q * e

            score = dot / (query_norm * emb_norm)

            retrieved_chunks.append(
                RetrievedChunk(
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                )
            )

        retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
        return retrieved_chunks[:top_k]