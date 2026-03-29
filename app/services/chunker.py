from app.schemas.rag import Chunk, Document


def split_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须 > 0")
    if overlap < 0:
        raise ValueError("overlap 不能 < 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须 < chunk_size")

    chunks = []
    text_length = len(text)
    start = 0
    while start < text_length:
        ideal_end = min(start + chunk_size, text_length)
        end = ideal_end

        # 如果还没到文本末尾，尽量往前找空白字符，避免把单词切开
        if end < text_length:
            min_end = min(start + max(1, chunk_size // 2), text_length)

            while end > min_end and not text[end - 1].isspace():
                end -= 1

            # 如果退太多都没找到合适空白，就退回原来的硬切位置
            if end == min_end and not text[end - 1].isspace():
                end = ideal_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # 下一段从 end - overlap 开始，保留重叠区
        next_start = max(0, end - overlap)

        # 跳过开头多余空白，避免 chunk 开头是空格或换行
        while next_start < text_length and text[next_start].isspace():
            next_start += 1

        # 防止极端情况下卡死
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


def split_documents(documents: list[Document], chunk_size: int = 200, overlap: int = 50) -> list[Chunk]:
    all_chunks = []

    for doc in documents:
        chunks = split_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            chunk_1=Chunk(
                source= doc.source,
                chunk_id=idx,
                text=chunk,
                doc_id=doc.doc_id
            )
            all_chunks.append(
              chunk_1
            )

    return all_chunks


