from sqlalchemy.orm import Session, selectinload

from app.schemas.rag import Chunk


def creat_chunks(db:Session,chunks:Chunk):
    ...

