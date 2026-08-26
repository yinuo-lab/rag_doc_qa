
from pydantic import BaseModel, Field

from app.schemas.rag import RetrievedChunk, SourceItem
from app.services.rag_pipeline import SIMILARITY_THRESHOLD
from app.services.retriever import Retriever


class RetrievalToolInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=10)


class RetrievalToolOutput(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    sources: list[SourceItem]
    has_enough_context: bool
    message: str
    error: str | None = None



class RetrievalTool:

    def __init__(
        self,
        retriever: Retriever,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.retriever=retriever
        self.threshold=threshold  #注入工具检索器和最低的要求top_k达到的分数

    def run(self, request: RetrievalToolInput) -> RetrievalToolOutput:
        query = request.query.strip()
        if not query:  #如果没有问题，返回问题不能为空，以及变量has_enough_context，是判断数据库内是否有跟问题相关性足够的文章
            return RetrievalToolOutput(
                query="",
                chunks=[],
                sources=[],
                has_enough_context=False,
                message="问题不能为空。",
                error="empty_query",
            )

        try:
            candidates = self.retriever.retrieve(query, top_k=request.top_k)#返回检索结果
        except Exception as exc:#检索失败
            return RetrievalToolOutput(
                query=query,
                chunks=[],
                sources=[],
                has_enough_context=False,
                message="检索工具暂时不可用。",
                error=type(exc).__name__,
            )

        has_enough_context = bool(candidates) and (#需要检索成功以及最大的top_k分数大于阈值
                candidates[0].score >= self.threshold
        )
        if not has_enough_context:#检索失败活着阈值不到
            return RetrievalToolOutput(
                query=query,
                chunks=[],
                sources=[],
                has_enough_context=False,
                message="知识库中没有找到足够相关的资料，无法回答该问题。",
            )

        sources = [#来源的文章信息
            SourceItem(
                source=chunk.source,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
            )
            for chunk in candidates
        ]
        return RetrievalToolOutput(#返回查找到的资料
            query=query,
            chunks=candidates,
            sources=sources,
            has_enough_context=True,
            message="已找到相关资料。",
        )
