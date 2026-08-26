from fastapi import APIRouter, HTTPException

from app.core.pipeline import pipeline
from app.schemas.agent import AgentAskRequest, AgentAskResponse
from app.services.agent_graph import DocumentAgent, OpenAICompatibleAgentModel
from app.services.retrieval_tool import RetrievalTool

router = APIRouter(prefix="/agent", tags=["agent"])

agent = DocumentAgent(
    retrieval_tool=RetrievalTool(pipeline.retriever),
    model=OpenAICompatibleAgentModel(),
)


@router.post("/ask", response_model=AgentAskResponse)
async def ask_agent(request: AgentAskRequest) -> AgentAskResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    return agent.ask(query=query, max_steps=request.max_steps)
