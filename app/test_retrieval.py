from app.services.loader import load_documents
from app.services.chunker import split_documents
from app.services.embedder import BiEncoderEmbedder
from app.services.vector_store import InMemoryVectorStore
from app.services.retriever import Retriever


def main():
    # 1. 读取文档
    documents = load_documents("app/data")
    print("=== documents ===")
    for doc in documents:
        print(f"source={doc.source}, text_len={len(doc.text)}")

    # 2. 切分 chunks
    chunks = split_documents(documents, chunk_size=80, overlap=20)
    print("\n=== chunks ===")
    for chunk in chunks:
        print(f"source={chunk.source}, chunk_id={chunk.chunk_id}, text={chunk.text!r}")

    # 3. 初始化 embedder / vector store / retriever
    embedder = BiEncoderEmbedder(dim=64)
    vector_store = InMemoryVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    # 4. 为 chunks 生成 embeddings
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    # 5. 建库
    vector_store.add_chunks(chunks, embeddings)
    print(f"\n=== index built ===")
    print(f"stored records: {len(vector_store.records)}")

    # 6. 测试 query
    query = "What are the steps of a typical RAG pipeline?"
    print(f"\n=== query ===\n{query}")

    results = retriever.retrieve(query, top_k=3)
    #sgkrutrd
    #测试脚本git commit -m "
    # 7. 打印检索结果
    print("\n=== retrieved chunks ===")
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] score={item.score:.4f}")
        print(f"source={item.source}, chunk_id={item.chunk_id}")
        print(f"text={item.text}")
        print("-" * 50)


if __name__ == "__main__":
    main()