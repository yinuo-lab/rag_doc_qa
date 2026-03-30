
from fastapi import FastAPI
from app.api.routes.ask import router as ask_router
app = FastAPI(title="RAG Doc QA")

app.include_router(ask_router)