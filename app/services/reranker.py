from app.schemas.rag import RetrievedChunk
from app.services.embedder import BiEncoderEmbedder
from app.services.vector_store import InMemoryVectorStore


class Reranker:
    def __init__(
        self
    ):
        self.embedder = BiEncoderEmbedder()
        self.vector_store = InMemoryVectorStore()

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
        ...
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        query_embedding = self.embedder.embed_text(query)
        if not query_embedding:
            raise RuntimeError("failed to create query embedding")
        for chunk in chunks:
            embedding=self.embedder.embed_text(chunk.text)
            s=0
            if len(embedding)!=len(query_embedding):
                continue
            for i in range(len(embedding)):
                if embedding[i]==query_embedding[i] and embedding[i]!=0:
                    s+=1
            chunk.score=s
        chunks=sorted(chunks,key=lambda x:x.score,reverse=True)
        if len(chunks)<top_n:
            return chunks
        return chunks[:top_n]

