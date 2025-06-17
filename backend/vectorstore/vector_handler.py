import os
from typing import List
import chromadb

from vectorstorage.chroma_config import CHROMA_SETTINGS
from vectorstorage.embedder import Embedder


class VectorHandler:
    def __init__(self, collection_name: str = "medical_faq_collection"):
        self.client = chromadb.Client(CHROMA_SETTINGS)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = Embedder()

    def add_documents(self, ids: List[str], texts: List[str], metadata: List[dict]):
        embeddings = self.embedder.embed(texts)
        self.collection.add(
            documents=texts,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata
        )

    def query(self, query_text: str, top_k: int = 5):
        embedding = self.embedder.embed([query_text])[0]
        return self.collection.query(query_embeddings=[embedding], n_results=top_k)

    def persist(self):
        self.client.persist()
