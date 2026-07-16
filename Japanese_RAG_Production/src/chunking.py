"""Japanese-aware text chunking module."""

from typing import List, Dict   # ← Add this line
from typing import List
import re
from src.config import config


class JapaneseChunker:
    """Handles intelligent chunking for Japanese text."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    def chunk_text(self, text: str, filename: str = "unknown") -> List[Dict]:
        """Split Japanese text into overlapping chunks."""
        if not text:
            return []
        
        # Simple but effective Japanese chunking
        # Split on sentence boundaries where possible
        sentences = re.split(r'([。！？])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": {
                            "filename": filename,
                            "chunk_id": len(chunks)
                        }
                    })
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    "filename": filename,
                    "chunk_id": len(chunks)
                }
            })
        
        print(f"✅ Created {len(chunks)} chunks from {filename}")
        return chunks


# For testing
if __name__ == "__main__":
    chunker = JapaneseChunker()
    sample_text = "これはテスト文書です。日本語のテキストを適切にチャンク化する必要があります。もう一つの文。"
    chunks = chunker.chunk_text(sample_text, "test.txt")
    print(f"Sample chunks created: {len(chunks)}")