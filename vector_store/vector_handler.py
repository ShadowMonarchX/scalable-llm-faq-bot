import os
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from vector_store.chroma_config import CHROMA_SETTINGS
from vector_store.embedder import Embedder

# Collection name
COLLECTION_NAME = "faq_documents"

class VectorStoreHandler:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(settings=CHROMA_SETTINGS)
        self.embedder = Embedder()
        self.collection = self._load_or_create_collection(COLLECTION_NAME)

    def _load_or_create_collection(self, name: str):
        if name in self.client.list_collections():
            return self.client.get_collection(name=name)
        return self.client.create_collection(name=name)

    def add_documents(self, documents: List[str], metadatas: List[dict], ids: List[str]):
        embeddings = self.embedder.embed_texts(documents)
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

    def query(self, query: str, top_k: int = 3):
        query_embedding = self.embedder.embed_text(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results.get("documents", [[]])[0]
