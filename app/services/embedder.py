import hashlib


class BiEncoderEmbedder:
    def __init__(self, dim: int = 64):
        self.dim=dim

    def _tokenize(self, text: str) -> list[str]:
        text=text.strip()
        text=text.lower()
        return text.split()
    def embed_text(self, text: str) -> list[float]:
        embed=[0]*self.dim
        texts=self._tokenize(text)
        for i in texts:
            embed[self._stable_hash(i)]+=1
        return embed
    def _stable_hash(self, token: str) -> int:
        return int(hashlib.sha256(token.encode("utf-8")).hexdigest(),16)%self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeds=[]
        for text in texts:
            embeds.append(self.embed_text(text))
        return embeds