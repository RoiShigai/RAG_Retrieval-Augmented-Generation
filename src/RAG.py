from .Indexor.Chunker.Chunk import Chunk
from .Indexor.Indexor import Indexor
from .Algorithm import BM25Index
from .DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from pathlib import Path
from typing import List


class RagError(Exception):
    def __init__(self, err: str) -> None:
        super().__init__(err)


class RAG:
    """
        RAG Class Definition:

        Core part of the RAG project. This class contain the method
            to run every part of the pipeline and handle the given database.
    """

    def __init__(self, database: Path) -> None:
        """ Init Method of the RAG Main Class """
        self.__bm25: BM25Index = BM25Index()
        self.__database = DataBaseHandler(database)

    def index(
            self,
            corpus: Path,
            max_chunk_size: int = 2000) -> None:
        """
            Indexing method for the RAG. This method will launch
                the indexor pipeline on a given corpus path and
                will store the result into the RAG database.

            Parameters:
                corpus: Path | The path to the corpus that will be indexed
                max_chunk_size: int | The maximum size* of text a Chunk
                    refers to

            *If not precise the max chunk size will be set to 2000 char
        """
        indexor: Indexor = Indexor(self.__database, max_chunk_size)
        chunk_corpus: List[Chunk] = indexor.generate_chunks(corpus)

        inverted_index = self.__bm25.create_index(chunk_corpus)
        self.__database.store_bm25_index(inverted_index)
        print("End of Indexing...")

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

    def debug_db(self) -> None:
        chunks_db = self.__database.get_all_chunks()
        reverse_index_db = self.__database.load_bm25_index()
        print("[DATABASE CHUNK DEBUGGING]")
        for chunks in chunks_db:
            chunks.debug_chunk()
        print("[DATABASE INDEX DEBUGGING]")
        for k, v in reverse_index_db.items():
            print("key: {k}: value: {v}")
