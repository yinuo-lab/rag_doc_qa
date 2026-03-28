from app.schemas.rag import RetrievedChunk
from app.services.embedder import BiEncoderEmbedder
from app.services.vector_store import InMemoryVectorStore


class Retriever:
    def __init__(
        self,
        embedder: BiEncoderEmbedder,
        vector_store: InMemoryVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        query_embedding = self.embedder.embed_text(query)
        if not query_embedding:
            raise RuntimeError("failed to create query embedding")

        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return retrieved_chunks