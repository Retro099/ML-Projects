"""Embedding and vector store management."""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import NotFoundError
from src.config import config
import uuid


class EmbeddingStore:
    """Handles embeddings and ChromaDB vector storage."""

    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
        self.collection_name = "japanese_docs"

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            # Collection does not exist yet
            self.collection = self.client.create_collection(name=self.collection_name)

        print(f"✅ Embedding model loaded: {config.EMBEDDING_MODEL}")

    def add_documents(self, chunks: List[Dict]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]

        # Generate unique IDs to avoid conflicts
        ids = [str(uuid.uuid4()) for _ in chunks]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(chunks)} chunks to vector store")

    def query(self, query_text: str, top_k: int = None) -> Dict:
        """Query the vector store using manual embedding."""
        top_k = top_k or config.TOP_K

        query_embedding = self.embedding_model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results