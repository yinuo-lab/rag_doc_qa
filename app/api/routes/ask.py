from fastapi import APIRouter, HTTPException

from app.core.pipeline import pipeline
from app.schemas.rag import AskRequest, AskResponse

router = APIRouter(prefix="", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    print("raw query =", repr(request.query))

    query = request.query.strip()
    print("stripped query =", repr(query))

    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    response = pipeline.ask(query)
    return response