from app.schemas.rag import SourceItem, AskResponse
from app.services.llm import LLMClient
from app.services.prompt_builder import build_prompt
from app.services.reranker import Reranker
from app.services.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever:Retriever, reranker:Reranker, llm_client:LLMClient):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client

    def ask(self, query: str) -> AskResponse:
        retrieved = self.retriever.retrieve(query, top_k=10)
        reranked = self.reranker.rerank(query, retrieved, top_n=3)
        prompt = build_prompt(query, reranked)
        answer = self.llm_client.generate(prompt)
        sources=[]
        for i in reranked:
            sources.append(SourceItem(source=i.source,chunk_id=i.chunk_id))
        return AskResponse(answer=answer,sources=sources)