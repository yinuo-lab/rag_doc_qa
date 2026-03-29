
class LLMClient:
    def generate(self, prompt: str) -> str:
        ...
        if prompt=='':
            return "没有找到答案"
        else:
            return "This is a mock answer."
