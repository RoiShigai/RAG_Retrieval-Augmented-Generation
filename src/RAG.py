class RagError(Exception):
    def __init__(self, err: str) -> None:
        super().__init__(err)


class RAG:
    def index(
            self,
            max_chunk_size: int) -> None:
        ...

    def search(
            self,
            query: str,
            k: int) -> None:
        ...

    def search_dataset(
            self,
            dataset_path: str,
            k: int, save_directory: str) -> None:
        ...

    def answer(
            self,
            query: str,
            k: int) -> None:
        ...

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str) -> None:
        ...

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str) -> None:
        ...
