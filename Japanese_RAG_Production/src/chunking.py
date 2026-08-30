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
                overlap = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else ""
                current_chunk = overlap + sentence
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
    sample_text = (
        "これはテスト文書です。日本語のテキストを適切にチャンク化する必要があります。"
        "もう一つの文。重なりを確認するために文章を長くします。"
        "楽天の2025年度の連結Non-GAAP営業利益は1,063億円でした。"
        "売上収益は前年比9.5%増の2兆4,966億円となりました。"
        "フィンテックセグメントの増収に加え全セグメントで増収増益を達成しました。"
        "社内ではRakuten AI for Rakuteniansが日常業務に定着しています。"
        "営業チームではメール対応時間を54%、資料作成時間を48%削減しました。"
        "日本語に最適化したオープン大規模言語モデルRakuten AI 7Bを公開しました。"
        "検索ツールへのAI組み込みによりクリック率と広告収益が向上しました。"
        "トリプル20はマーケティング、オペレーション、クライアント効率の改善目標です。"
        "以上の文は重なり確認用であり、実際のPDFを再取り込みしてはいません。"
    ) * 3
    chunks = chunker.chunk_text(sample_text, "test.txt")
    print(f"Sample chunks created: {len(chunks)}")
    if len(chunks) >= 2:
        print("CHUNK1 END:", chunks[0]["content"][-60:])
        print("CHUNK2 START:", chunks[1]["content"][:60:])