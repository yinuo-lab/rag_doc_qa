from pathlib import Path

from app.services.loader import load_documents
from app.services.chunker import split_documents
from app.services.embedder import BiEncoderEmbedder
from app.services.vector_store import InMemoryVectorStore
from app.services.retriever import Retriever
from app.services.reranker import Reranker
from app.services.llm import LLMClient
from app.services.rag_pipeline import RAGPipeline
from functools import lru_cache

def build_pipeline() -> RAGPipeline:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    documents = load_documents(str(data_dir))
    chunks = split_documents(documents, chunk_size=80, overlap=20)

    embedder = BiEncoderEmbedder(dim=64)
    vector_store = InMemoryVectorStore()

    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    vector_store.add_chunks(chunks, embeddings)

    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    reranker = Reranker()
    llm_client = LLMClient()

    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
    )

pipeline = build_pipeline()