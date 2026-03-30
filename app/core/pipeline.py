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
from app.services.index_store import  sync_index
def build_pipeline() -> RAGPipeline:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    INDEX_DIR=BASE_DIR/'index'
    chunks =sync_index(str(DATA_DIR),str(INDEX_DIR))

    embedder = BiEncoderEmbedder(model="qwen3-embedding:0.6b-fp16")
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