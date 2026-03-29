from pydantic import BaseModel


class Document(BaseModel):
    source: str
    text: str
    doc_id: str

class Chunk(BaseModel):
    doc_id: str
    source: str
    chunk_id: int
    text: str
class SourceItem(BaseModel):
    source: str
    doc_id:str
    chunk_id: int

class RetrievedChunk(BaseModel):
    source: str
    chunk_id: int
    text: str
    score: float
    doc_id:str
class DocRegistryItem(BaseModel):
    doc_id:str
    source:str
    content_hash:str
    chunk_count:int
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

