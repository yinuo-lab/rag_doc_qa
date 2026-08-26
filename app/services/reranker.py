import re

from app.schemas.rag import RetrievedChunk


class Reranker:
    #轻量级混合重排

    @staticmethod
    def _features(text: str) -> set[str]:#提取文本的词汇特征，包括英文单词和字符二元组，用来计算词汇重叠。
        normalized = re.sub(r"\s+", "", text.lower())#     # 去掉所有空白，并转小写
        features = set(re.findall(r"[a-z0-9_]+", normalized))# # 提取类似英文单词、数字、下划线组成的 token
        features.update(
            normalized[index : index + 2]#  # 加入字符级 2-gram，辅助中文或无空格文本的匹配
            for index in range(max(0, len(normalized) - 1))
        )
        return features

    def _lexical_score(self, query: str, text: str) -> float:
        #计算 query 和 chunk.text 的词汇重叠度。
        query_features = self._features(query)
        text_features = self._features(text)
        if not query_features or not text_features:
            return 0.0
        return len(query_features & text_features) / len(query_features)#返回重叠个数

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
        query = query.strip()#重排序
        if not query:
            raise ValueError("query cannot be empty")

        reranked = []
        for chunk in chunks:
            vector_score = max(0.0, min(1.0, (chunk.score + 1.0) / 2.0))#归一化
            lexical_score = self._lexical_score(query, chunk.text)
            combined_score = 0.85 * vector_score + 0.15 * lexical_score# # 加权融合
            reranked.append(chunk.model_copy(update={"score": combined_score}))## 复制 chunk，并用融合后的分数覆盖原 score

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_n]

