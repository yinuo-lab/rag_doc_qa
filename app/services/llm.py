# app/services/llm.py
from openai import OpenAI

from app.core.config import settings


class LLMClient:
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        if settings.openai_base_url:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        else:
            self.client = OpenAI(api_key=settings.openai_api_key)

        self.model = settings.openai_model

    def generate(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "没有找到足够的资料来回答这个问题。"

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        answer = (response.output_text or "").strip()
        if not answer:
            return "模型没有返回可见文本。"
        return answer
