from app.schemas.rag import RetrievedChunk

class Reranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
        ...