
from fastapi import FastAPI
from app.api.routes.ask import router as ask_router
from app.api.routes.agent import router as agent_router

app = FastAPI(title="RAG Doc QA")
app.include_router(ask_router)
app.include_router(agent_router)
