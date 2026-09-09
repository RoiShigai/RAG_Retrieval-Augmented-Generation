from .RAG import RAG
from pathlib import Path

rag = RAG(Path("data/processed/database.db"))
#rag.index(Path("vllm-0.10.1/"))
#print("indexing...")
res = rag.search("where are the test ?", 5)
for r in res:
    print(r)
#rag.debug_db()
