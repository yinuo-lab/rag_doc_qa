# app/services/rag_pipeline.py
# 主流程编排：Retrieve → 阈值过滤 → Rerank → Prompt → Generate → 返回 answer + sources

from app.schemas.rag import SourceItem, AskResponse
from app.services.llm import LLMClient
from app.services.prompt_builder import build_prompt
from app.services.reranker import Reranker
from app.services.retriever import Retriever

# 相似度阈值：检索出的最高余弦相似度低于它，就认为资料不足，直接拒答
SIMILARITY_THRESHOLD = 0.35


class RAGPipeline:
    def __init__(self, retriever: Retriever, reranker: Reranker, llm_client: LLMClient):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client

    def ask(self, query: str) -> AskResponse:
        # 第一步：召回 top-10 候选
        retrieved = self.retriever.retrieve(query, top_k=10)

        # 第二步：阈值过滤——最高分都低于阈值，说明知识库里没有相关资料
        if not retrieved or retrieved[0].score < SIMILARITY_THRESHOLD:
            return AskResponse(
                answer="知识库中没有找到足够相关的资料，无法回答该问题。",
                sources=[],
            )

        # 第三步：重排，精选最相关的 1 条
        reranked = self.reranker.rerank(query, retrieved, top_n=1)

        # 第四步：拼 prompt
        prompt = build_prompt(query, reranked)

        # 第五步：LLM 生成答案
        answer = self.llm_client.generate(prompt)

        # 第六步：构造 sources（按 doc_id 去重）
        filtered_chunks = []
        pasted = {}
        for item in reranked:
            if item.doc_id in pasted:
                continue
            pasted[item.doc_id] = 1
            filtered_chunks.append(item)

        sources = []
        for item in filtered_chunks:
            sources.append(
                SourceItem(
                    source=item.source,
                    doc_id=item.doc_id,
                    chunk_id=item.chunk_id,
                )
            )
        return AskResponse(answer=answer, sources=sources)
