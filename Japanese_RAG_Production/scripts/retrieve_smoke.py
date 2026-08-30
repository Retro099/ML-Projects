"""Retrieval-only smoke test. Does not call Groq."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import Retriever

QUESTIONS = [
    "楽天の2025年度の連結Non-GAAP営業利益はいくらでしたか？",
    "楽天はAIをどのように活用していますか？",
]

def main():
    retriever = Retriever()
    for q in QUESTIONS:
        print("\n" + "=" * 60)
        print("Q:", q)
        chunks = retriever.retrieve(q, top_k=5)
        print("n_chunks:", len(chunks))
        for i, ch in enumerate(chunks, 1):
            name = ch.get("metadata", {}).get("filename", "?")
            dist = ch.get("distance")
            text = ch.get("content", "").replace("\n", " ")[:180]
            print(f"  [{i}] {name} dist={dist}")
            print(f"      {text}")

if __name__ == "__main__":
    main()