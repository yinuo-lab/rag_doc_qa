from app.schemas.rag import RetrievedChunk


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    if not chunks:
        return ""

    parts = []
    parts.append("你是文档问答助手，只能基于提供的资料回答；如果资料不足，就明确回答“不知道”。")
    parts.append("")

    for idx, chunk in enumerate(chunks, start=1):
        parts.append(f"资料{idx}（source={chunk.source}, chunk_id={chunk.chunk_id}）:")
        parts.append(chunk.text)
        parts.append("")

    parts.append(f"问题：{query}")

    return "\n".join(parts)