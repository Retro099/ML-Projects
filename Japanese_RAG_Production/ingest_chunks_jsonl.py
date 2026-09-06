"""Build a fresh chroma_db from Colab chunks_v2_overlap.jsonl.

Run from the Japanese_RAG_Production folder, inside venv.
Forces CPU so the 940MX is not used.
Does not call Groq.
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent
if (ROOT / "src").is_dir():
    PROJECT = ROOT
else:
    PROJECT = ROOT
    # allow dropping this file in scripts/
    if PROJECT.name == "scripts":
        PROJECT = PROJECT.parent

sys.path.insert(0, str(PROJECT))

from src.config import config
from src.embedding_store import EmbeddingStore

JSONL = PROJECT / "chunks_v2_overlap.jsonl"
NEW_DB = PROJECT / "chroma_db_v2_overlap"


def main():
    if not JSONL.exists():
        raise FileNotFoundError(f"Put the Colab file here first: {JSONL}")

    chunks = []
    with JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print("loaded chunks:", len(chunks))
    if len(chunks) != 713:
        print("WARNING: expected 713 from the Colab export")

    names = {}
    for c in chunks:
        fn = c.get("metadata", {}).get("filename", "?")
        names[fn] = names.get(fn, 0) + 1
    for fn, n in names.items():
        print(f"  {fn}: {n}")

    # Do not touch locked chroma_db. Write a new folder.
    if NEW_DB.exists():
        shutil.rmtree(NEW_DB)
    config.CHROMA_DB_PATH = NEW_DB

    store = EmbeddingStore()
    BATCH = 32
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        store.add_documents(batch)
        print(f"added {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print("collection count:", store.collection.count())
    print("done:", NEW_DB)
    print("old index left untouched:", PROJECT / "chroma_db")


if __name__ == "__main__":
    main()
