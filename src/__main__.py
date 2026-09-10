from .RAG import RAG
from pathlib import Path
import fire
import sys


def index(max_chunk_size):
    """
        Indexing function called by Fire plugin
    """
    rag = RAG(Path("data/processed/database.db"))
    rag.index(max_chunk_size=max_chunk_size)


def search(query: str, k: int) -> None:
    """
        Search the k most relevant chunks for a UserQuery
        This function is meant to be called by the Fire plugin
    """
    rag = RAG(Path("data/processed/database.db"))
    res = rag.search(query, k)
    print("result:")
    for r in res:
        print(r)


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    """ Search dataset function called by Fire plugin """
    rag = RAG(Path("data/processed/database.db"))
    rag.search_dataset(
        Path(dataset_path),
        k,
        Path(save_directory)
    )


def answer(query: str, k: int) -> None:
    """
        Answer a single query using the retrieved context
        This function is meant to be called with Fire plugin
    """
    rag = RAG(Path("data/processed/database.db"))
    rag.answer(query, k)


def answer_dataset(
        student_search_results_path: str,
        save_directory: str) -> None:
    """
        Generate answers for a dataset,
        Produce StudentSearchResultsAndAnswers JSON file
    """
    rag = RAG(Path("data/processed/database.db"))
    rag.answer_dataset(
        Path(student_search_results_path),
        Path(save_directory)
    )


def evaluate(
        student_search_results_path: str,
        dataset_path: str) -> None:
    ...

def test() -> None:
    rag = RAG(Path("data/processed/database.db"))
    #rag.index(Path("vllm-0.10.1/"))
    #print("indexing...")
    res = rag.search("How to install on linux machine ?", 5)
    for r in res:
        print(r)
    #rag.debug_db()


try:
    fire.Fire()
except (Exception, KeyboardInterrupt) as e:
    print(f"Catch {type(e).__name__}: {e}")
    sys.exit()
