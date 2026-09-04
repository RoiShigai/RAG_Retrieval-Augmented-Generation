from .Indexor.Chunker.Chunk import Chunk
from .Indexor.Indexor import Indexor
from .Algorithm import BM25Index
from .DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from pathlib import Path
from typing import List, cast

from .Algorithm.Match import ChunkKey
from .Model import MinimalSource


class RagError(Exception):
    def __init__(self, err: str) -> None:
        super().__init__(err)


class RAG:
    """
        RAG Class Definition:

        Core part of the RAG project. This class contain the method
            to run every part of the pipeline and handle the given database.
    """

    def __init__(
            self,
            database: Path,
            corpus: Path = Path("vllm-0.10.1"),
            ) -> None:
        """ Init Method of the RAG Main Class """
        self.__corpus = corpus.resolve()
        if not self.__corpus.is_dir():
            raise NotADirectoryError(self.__corpus)
        self.__database = DataBaseHandler(database)
        self.__bm25 = self.__load_bm25()

    def __load_bm25(self) -> BM25Index:
        """Load the persisted BM25 index."""
        index, stats, lengths, count = self.__database.load_bm25_data()
        return BM25Index.from_persisted(
            cast(dict[str, dict[ChunkKey, int]], index),
            stats,
            cast(dict[ChunkKey, int], lengths),
            count,
        )

    def __synchronize(self) -> None:
        """Synchronize the corpus and reload BM25 when it changes."""
        changed = self.__database.synchronize_corpus(self.__corpus)
        if changed:
            chunks = Indexor(self.__database).generate_chunks(self.__corpus)
            created = BM25Index().create_index(chunks)
            self.__database.store_bm25_index(
                cast(dict[str, dict[tuple[str, int], int]], created)
            )
            self.__bm25 = self.__load_bm25()

    def index(
            self,
            corpus: Path | None = None,
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
        selected_corpus = self.__corpus if corpus is None else corpus.resolve()
        indexor: Indexor = Indexor(self.__database, max_chunk_size)
        indexor.generate_chunks(selected_corpus)
        chunk_corpus: List[Chunk] = self.__database.get_all_chunks()

        inverted_index = self.__bm25.create_index(chunk_corpus)
        self.__database.store_bm25_index(
            cast(dict[str, dict[tuple[str, int], int]], inverted_index)
        )
        print("End of Indexing...")

    def search(
            self,
            query: str,
            k: int) -> list[MinimalSource]:
        """Return source locations matching the query."""
        self.__synchronize()
        matches = self.__bm25.search(query.split(), k)
        sources: list[MinimalSource] = []
        for chunk_key, _score in matches:
            if not isinstance(chunk_key, tuple):
                continue
            chunk = self.__database.get_chunk(*chunk_key)
            sources.append(MinimalSource.model_construct(
                file_path=str(chunk.file_path),
                first_character_index=chunk.start,
                last_character_index=chunk.end,
            ))
        return sources

    def search_dataset(
            self,
            dataset_path: str,
            k: int, save_directory: str) -> None:
        ...

    def answer(
            self,
            query: str,
            k: int) -> None:
        """Synchronize the corpus before generating an answer."""
        self.__synchronize()

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
        """ Debug function to check what is stored into the db """
        reverse_index_db = self.__database.load_bm25_index()
#       print("[DATABASE CHUNK DEBUGGING]")
#       for chunks in chunks_db:
#           chunks.debug_chunk()
        print("[DATABASE INDEX DEBUGGING]")
        for k, v in reverse_index_db.items():
            print(f"key: {k}: value: {v}")
