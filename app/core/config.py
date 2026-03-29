# app/core/config.py
from pydantic import BaseModel
import os


class Settings(BaseModel):
    openai_api_key: str
    openai_model: str = "gpt-5.4"
    openai_base_url: str | None = None


settings = Settings(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
    openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
)