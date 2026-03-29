from app.services.loader import load_documents
from app.services.chunker import split_documents
from app.services.embedder import BiEncoderEmbedder
from app.services.vector_store import InMemoryVectorStore
from app.services.retriever import Retriever
from app.services.reranker import Reranker
from app.services.llm import LLMClient
from app.services.rag_pipeline import RAGPipeline


def main():
    documents = load_documents("app/data")
    chunks = split_documents(documents, chunk_size=80, overlap=20)

    embedder = BiEncoderEmbedder(dim=64)
    vector_store = InMemoryVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    reranker = Reranker()
    llm_client = LLMClient()
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
    )

    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    vector_store.add_chunks(chunks, embeddings)

    query = "What are the steps of a typical RAG pipeline?"
    response = pipeline.ask(query)

    print("=== answer ===")
    print(response.answer)

    print("\n=== sources ===")
    for source in response.sources:
        print(source)


if __name__ == "__main__":
    main()