from Indexor.Indexor import Indexor
from Algorithm import DatabaseBM25Index
from DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from DataHandler import CacheHandler
from pathlib import Path
from typing import List
from Model import (
    MinimalSource, MinimalSearchResults, MinimalAnswer, UnansweredQuestion,
)
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
            cache_directory: Path = Path("data/cache"),
            ) -> None:
        """ Init Method of the RAG Main Class """
        self.start = time.perf_counter()
        self.__corpus = corpus.resolve()
        if not self.__corpus.is_dir():
            raise NotADirectoryError(self.__corpus)
        self.__database = DataBaseHandler(database)
        self.__bm25 = DatabaseBM25Index(self.__database)
        self.__cache = CacheHandler(cache_directory)
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
        indexor.generate_chunks(selected_corpus, include_existing=False)
        self.__database.refresh_bm25_metadata()
        print("End of Indexing...")

    def search(
            self,
            query: str,
            k: int) -> list[MinimalSource]:
        """Return source locations matching the query."""
        database_version = self.__database.get_database_version()
        print(f"database version: {time.perf_counter() - self.start}")
        cached = self.__cache.get_search_result(query, k, database_version)
        print(f"cache query: {time.perf_counter() - self.start}")
        if cached is not None:
            return cached
        matches = self.__bm25.search(tokenize_text(query), k)
        print(f"BM25 matches: {time.perf_counter() - self.start}")
        sources: list[MinimalSource] = []
        for chunk_key, _score in matches:
            if not isinstance(chunk_key, tuple):
                continue
            chunk = self.__database.get_chunk(*chunk_key)
            print(f"load chunk: {time.perf_counter() - self.start}")
            sources.append(MinimalSource.model_construct(
                file_path=str(chunk.file_path),
                first_character_index=chunk.start,
                last_character_index=chunk.end,
            ))
        self.__cache.store_search_result(
            query, k, database_version, sources
        )
        print(f"cache storing: {time.perf_counter() - self.start}")
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
        print("BITE")
        dataset_file = load_json_file(dataset_path)
        print(dataset_file)
        dataset: List[UnansweredQuestion] = [
                UnansweredQuestion.model_construct(
                    question=q["question"], question_id=q["question_id"]
                    )
                for q in dataset_file["rag_questions"]
            ]
        print(f"data set construct: {time.perf_counter() - self.start}")

        for question in dataset:
            sources = self.search(question.question, k)
            print(f"question: {time.perf_counter() - self.start}")
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
            k: int = TOP_K) -> MinimalAnswer:
        """
            Generate an Answer to a User Query with SLM using
                the contest retrieved for this question.
        """
        database_version = self.__database.get_database_version()
        cached = self.__cache.get_answer(query, k, database_version)
        if cached is not None:
            print(cached.answer)
            return cached
        search_result = self.search(query, k)
        answer = self.__generate_answer(search_result, query)
        self.__cache.store_answer(query, k, database_version, answer)
        print(answer.answer)
        return answer

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str) -> None:
        """
            Answer to a whole dataset of question and
                store the response in a given directory

            Parameters:
                student_search_results_path: str | path to the questions
                    dataset
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
            Indexor(self.__database).generate_chunks(
                self.__corpus, include_existing=False
            )
            self.__database.refresh_bm25_metadata()
            self.__bm25 = DatabaseBM25Index(self.__database)

    def debug_db(self) -> None:
        """ Debug function to check what is stored into the db """
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
        generated_answer = self.__model.generate_response(search_result, query)
        return MinimalAnswer.model_construct(
            question_id=query,
            question=query,
            retrieved_sources=search_result,
            answer=generated_answer,
        )
