from enum import Enum
import hashlib
import json
from pathlib import Path

from Model import MinimalAnswer, MinimalSource

from .CacheDataModel import CacheDataModel


class CacheDataType(Enum):
    """Identify the kind of value stored in a cache entry."""

    ANSWER = "answer"
    SEARCH_RESULT = "search_result"


CachedDataType = CacheDataType


class CacheHandler:
    """Read and write versioned RAG results on the local filesystem."""

    def __init__(self, cache_directory: Path = Path("data/cache")) -> None:
        """Initialize the cache handler and its storage directory."""
        self.__cache_directory = cache_directory
        self.__cache_directory.mkdir(parents=True, exist_ok=True)

    def get_search_result(
            self, query: str, k: int, database_version: str
            ) -> list[MinimalSource] | None:
        """Return a current cached search result, if one exists."""
        entry = self.__load(
            query, k, database_version, CacheDataType.SEARCH_RESULT
        )
        if entry is None or entry.search_result is None:
            return None
        return entry.search_result

    def get_answer(
            self, query: str, k: int, database_version: str
            ) -> MinimalAnswer | None:
        """Return a current cached answer, if one exists."""
        entry = self.__load(query, k, database_version, CacheDataType.ANSWER)
        if entry is None or entry.answer is None:
            return None
        return entry.answer

    def store_search_result(
            self,
            query: str,
            k: int,
            database_version: str,
            search_result: list[MinimalSource],
            ) -> None:
        """Persist a search result with the database version it used."""
        self.__store(CacheDataModel(
            query=query,
            k=k,
            database_version=database_version,
            search_result=[source.model_dump() for source in search_result],
        ), CacheDataType.SEARCH_RESULT)

    def store_answer(
            self,
            query: str,
            k: int,
            database_version: str,
            answer: MinimalAnswer,
            ) -> None:
        """Persist an answer with the database version it used."""
        self.__store(CacheDataModel(
            query=query,
            k=k,
            database_version=database_version,
            answer=answer.model_dump(),
        ), CacheDataType.ANSWER)

    def __load(
            self,
            query: str,
            k: int,
            database_version: str,
            data_type: CacheDataType,
            ) -> CacheDataModel | None:
        """Load and validate one current cache entry."""
        try:
            with self.__cache_path(query, k, data_type).open(
                    "r", encoding="utf-8") as file:
                entry = CacheDataModel.model_validate(json.load(file))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if entry.query != query or entry.k != k:
            return None
        if entry.database_version != database_version:
            return None
        return entry

    def __store(
            self, entry: CacheDataModel, data_type: CacheDataType
            ) -> None:
        """Write one cache entry atomically enough for local use."""
        with self.__cache_path(entry.query, entry.k, data_type).open(
                "w", encoding="utf-8") as file:
            json.dump(entry.model_dump(mode="json"), file, indent=2)

    def __cache_path(
            self, query: str, k: int, data_type: CacheDataType
            ) -> Path:
        """Return the stable filename for a query and result size."""
        cache_key = f"{data_type.value}:{k}:{query}"
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.__cache_directory / f"{digest}.json"
