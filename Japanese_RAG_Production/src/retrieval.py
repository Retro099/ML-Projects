"""Retrieval logic from vector store."""

from typing import List, Dict
from src.embedding_store import EmbeddingStore
from src.config import config


class Retriever:
    """Handles querying the vector store and formatting results."""

    def __init__(self):
        self.store = EmbeddingStore()

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant chunks for a query."""
        top_k = top_k or config.TOP_K
        results = self.store.query(query, top_k=top_k)

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        retrieved = []
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc in enumerate(documents):
            retrieved.append({
                "content": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else None
            })

        return retrieved