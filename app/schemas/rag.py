from pydantic import BaseModel


class Document(BaseModel):
    source: str
    text: str


class Chunk(BaseModel):
    source: str
    chunk_id: int
    text: str


class RetrievedChunk(BaseModel):
    source: str
    chunk_id: int
    text: str
    score: float


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]