import re

from app.schemas.rag import RetrievedChunk


class Reranker:
    """Lightweight hybrid reranker using recall score plus lexical overlap.

    This is intentionally local and dependency-free. A Cross-Encoder remains a
    production upgrade because it can jointly model the query and each chunk.
    """

    @staticmethod
    def _features(text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text.lower())
        features = set(re.findall(r"[a-z0-9_]+", normalized))
        features.update(
            normalized[index : index + 2]
            for index in range(max(0, len(normalized) - 1))
        )
        return features

    def _lexical_score(self, query: str, text: str) -> float:
        query_features = self._features(query)
        text_features = self._features(text)
        if not query_features or not text_features:
            return 0.0
        return len(query_features & text_features) / len(query_features)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        reranked = []
        for chunk in chunks:
            vector_score = max(0.0, min(1.0, (chunk.score + 1.0) / 2.0))
            lexical_score = self._lexical_score(query, chunk.text)
            combined_score = 0.85 * vector_score + 0.15 * lexical_score
            reranked.append(chunk.model_copy(update={"score": combined_score}))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_n]

