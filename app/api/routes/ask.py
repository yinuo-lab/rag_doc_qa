from fastapi import APIRouter

from app.core.pipeline import pipeline
from app.schemas.rag import AskRequest, AskResponse

router = APIRouter(prefix="", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    response = pipeline.ask(request.query)
    return response