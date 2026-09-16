from Indexor.Chunker.Chunk import Chunk
from Indexor.Indexor import Indexor
from Algorithm import BM25Index, DatabaseBM25Index
from DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from pathlib import Path
from typing import List, cast
from Model import MinimalSource, MinimalSearchResults, MinimalAnswer, UnansweredQuestion
from Indexor.Chunker.Tokenizer.TokenNormalizer import tokenize_text
from Helper import load_json_file
from SLM import SLM
from llm_sdk import Small_LLM_Model
import time


TOP_K: int = 10


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
        start = time.perf_counter()
        self.__corpus = corpus.resolve()
        print(f"corpus resolve: {time.perf_counter() - start}")
        if not self.__corpus.is_dir():
            raise NotADirectoryError(self.__corpus)
        self.__database = DataBaseHandler(database)
        print(f"database creation: {time.perf_counter() - start}")
        self.__bm25 = DatabaseBM25Index(self.__database)
        print(f"loading bm25: {time.perf_counter() - start}")
        self.__model = None

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
        chunk_corpus: List[Chunk] = indexor.generate_chunks(
                    selected_corpus
                )
        #chunk_corpus: List[Chunk] = self.__database.get_all_chunks()

        inverted_index = BM25Index().create_index(chunk_corpus)
        self.__database.store_bm25_index(
            cast(dict[str, dict[tuple[str, int], int]], inverted_index)
        )
        print("End of Indexing...")

    def search(
            self,
            query: str,
            k: int) -> list[MinimalSource]:
        """Return source locations matching the query."""
        matches = self.__bm25.search(tokenize_text(query), k)
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
            dataset_path: Path,
            k: int,
            save_directory: Path = Path("data/dataset")
            ) -> List[MinimalSearchResults]:
        """
            Read a given dataset containing User request and
                search the top K chunks to answers each request and
                return a list of MinimalSearchResults

            Parameters:
                dataset_path: Path | the file path to the file containing
                    the user request
                k: int | the number of chunks retrieved to answer a question
                save_directory: Path | Path to the folder where

            Return:
                List of MinimalSearchResults object
        """
        sources: List[MinimalSource] = []
        answers: List[MinimalSearchResults] = []
        dataset: List[UnansweredQuestion] = [
                UnansweredQuestion.model_construct(question = q["question"])
                for q in load_json_file(dataset_path)
            ]

        for question in dataset:
            sources = self.search(question.question, k)
            answers.append(
                    MinimalSearchResults.model_construct(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources
                        )
                    )

        return answers

    def answer(
            self,
            query: str,
            k: int = TOP_K) -> None:
        """
            Generate an Answer to a User Query with SLM using
                the contest retrieved for this question.
        """
        search_result = self.search(query, k)
        answer = self.__generate_answer(search_result, query)
        print(answer)

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str) -> None:
        """
            Answer to a whole dataset of question and
                store the response in a given directory

            Parameters:
                student_search_results_path: str | path to the questions dataset
                save_directory: str | path to the directory
        """
        ...

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str) -> None:
        ...

    def __synchronize(self) -> None:
        """Synchronize the corpus and reload BM25 when it changes."""
        changed = self.__database.synchronize_corpus(self.__corpus)
        if changed:
            chunks = Indexor(self.__database).generate_chunks(self.__corpus)
            created = BM25Index().create_index(chunks)
            self.__database.store_bm25_index(
                cast(dict[str, dict[tuple[str, int], int]], created)
            )
            self.__bm25 = DatabaseBM25Index(self.__database)

    def debug_db(self) -> None:
        """ Debug function to check what is stored into the db """
#       print("[DATABASE CHUNK DEBUGGING]")
#       for chunks in chunks_db:
#           chunks.debug_chunk()
        print("[DATABASE INDEX DEBUGGING]")
        print("BM25 postings remain persisted in SQLite and are query-loaded.")

    def __generate_answer(
            self,
            search_result: List[MinimalSource],
            query: str) -> MinimalAnswer:
        """
            Generate a MinimalAnswer Object with the retrieved Source
                for a given query
        """
        if self.__model is None:
            self.__model = SLM(Small_LLM_Model())
        answer = self.__model.generate_response(search_result, query)
        return answer
