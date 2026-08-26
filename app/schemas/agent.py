from pydantic import BaseModel, Field

from app.schemas.rag import SourceItem


class AgentAskRequest(BaseModel):
    query: str
    max_steps: int = Field(default=2, ge=1, le=3)


class AgentAskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    steps: int
    stop_reason: str
