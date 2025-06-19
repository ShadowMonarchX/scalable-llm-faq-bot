from sentence_transformers import SentenceTransformer
from typing import List

# You can change to a different model if needed
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
