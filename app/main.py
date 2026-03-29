from contextlib import asynccontextmanager

from fastapi import FastAPI


from app.api.routes.ask import router as ask_router
from dotenv import load_dotenv

from app.db.base import Base
from app.db.session import engine

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    Base.metadata.create_all(bind=engine)
    yield
    # shutdown（目前不用写）

app = FastAPI(title="RAG Doc QA",lifespan=lifespan)

app.include_router(ask_router)