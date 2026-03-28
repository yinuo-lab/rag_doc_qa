from app.services.prompt_builder import build_prompt


class RAGPipeline:
    def __init__(self, retriever, reranker, llm_client):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client

    def ask(self, query: str) -> dict:
        retrieved = self.retriever.retrieve(query, top_k=10)
        reranked = self.reranker.rerank(query, retrieved, top_n=3)
        prompt = build_prompt(query, reranked)
        answer = self.llm_client.generate(prompt)

        return {
            "answer": answer,
            "sources": [
                {"source": c.source, "chunk_id": c.chunk_id}
                for c in reranked
            ],
        }