from openai import OpenAI


class LLMClient:
    def __init__(self, model: str = "qwen3:8b"):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "没有找到足够的资料来回答这个问题。"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                reasoning_effort="none",
                max_tokens=512,
            )

            answer = response.choices[0].message.content
            if answer is None:
                return "模型没有返回可见文本。"

            answer = answer.strip()
            if not answer:
                return "模型没有返回可见文本。"

            return answer

        except Exception as e:
            print("LLM call failed:", repr(e))
            raise