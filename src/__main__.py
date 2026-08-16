import sys
import fire
from .RAG import RAG

try:
    rag = RAG()
    fire.Fire(RAG)

except (Exception) as e:
    print(f"Catch: {e.__name__}: {e}")
    sys.exit()
