"""Document ingestion module - Supports both PDF and Markdown"""

from pathlib import Path
from typing import List, Dict
import pymupdf  # PyMuPDF


class DocumentIngestion:
    """Handles loading documents from PDF and Markdown files."""
    
    def load_pdf(self, file_path: Path) -> Dict:
        """Load text from a PDF file."""
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        return {
            "content": text,
            "filename": file_path.name,
            "file_type": "pdf"
        }
    
    def load_markdown(self, file_path: Path) -> Dict:
        """Load text from a Markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        return {
            "content": text,
            "filename": file_path.name,
            "file_type": "markdown"
        }
    
    def load_all_documents(self, folder_path: Path) -> List[Dict]:
        """Load all supported documents from a folder."""
        documents = []
        
        for file_path in folder_path.glob("*"):
            if file_path.suffix.lower() == ".pdf":
                try:
                    doc = self.load_pdf(file_path)
                    documents.append(doc)
                    print(f"✅ Loaded: {file_path.name} ({len(doc['content'])} characters)")
                except Exception as e:
                    print(f"❌ Failed to load {file_path.name}: {e}")
            
            elif file_path.suffix.lower() == ".md":
                try:
                    doc = self.load_markdown(file_path)
                    documents.append(doc)
                    print(f"✅ Loaded: {file_path.name} ({len(doc['content'])} characters)")
                except Exception as e:
                    print(f"❌ Failed to load {file_path.name}: {e}")
        
        print(f"\n📊 Total documents loaded: {len(documents)}")
        return documents