from .RAG import RAG
from pathlib import Path

rag = RAG(Path("data/processed/database.db"))
rag.index(Path("vllm-0.10.1/"))
print("indexing...")
#rag.debug_db()
