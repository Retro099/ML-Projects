"""Simple evaluation module for the RAG pipeline."""

from typing import List, Dict


class SimpleEvaluator:
    """Basic evaluation for retrieval quality."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_retrieval(self, query: str, retrieved_chunks: List[Dict], 
                           expected_keywords: List[str] = None) -> Dict:
        """
        Simple evaluation of retrieval results.
        - Checks how many retrieved chunks contain expected keywords (if provided)
        - Records basic stats
        """
        num_chunks = len(retrieved_chunks)
        total_chars = sum(len(chunk["content"]) for chunk in retrieved_chunks)
        
        keyword_hits = 0
        if expected_keywords:
            for chunk in retrieved_chunks:
                content_lower = chunk["content"].lower()
                if any(kw.lower() in content_lower for kw in expected_keywords):
                    keyword_hits += 1
        
        evaluation = {
            "query": query,
            "num_chunks_retrieved": num_chunks,
            "total_characters": total_chars,
            "keyword_hit_rate": keyword_hits / num_chunks if num_chunks > 0 else 0,
            "avg_chunk_length": total_chars / num_chunks if num_chunks > 0 else 0
        }
        
        self.results.append(evaluation)
        return evaluation
    
    def print_last_evaluation(self):
        if not self.results:
            print("No evaluations yet.")
            return
        
        last = self.results[-1]
        print("\n📊 Evaluation Results:")
        print(f"   Query: {last['query']}")
        print(f"   Chunks retrieved: {last['num_chunks_retrieved']}")
        print(f"   Keyword hit rate: {last['keyword_hit_rate']:.2%}")
        print(f"   Average chunk length: {last['avg_chunk_length']:.0f} chars")


# Quick test
if __name__ == "__main__":
    print("Evaluator module ready")