from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    
    # Works both locally and in Colab
    CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
    
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_SAMPLE = PROJECT_ROOT / "data" / "sample"
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    LLM_MODEL = "llama-3.3-70b-versatile"
    
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 5
    TEMPERATURE = 0.1
    MAX_TOKENS = 1024

config = Config()