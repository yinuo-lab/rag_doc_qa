from app.schemas.rag import RetrievedChunk

def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    ...