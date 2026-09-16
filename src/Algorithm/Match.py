from Indexor.Chunker.Chunk import Chunk
from typing import TYPE_CHECKING, List, Tuple
import heapq
import math

from Indexor.Chunker.Tokenizer.TokenNormalizer import tokenize_text

if TYPE_CHECKING:
    from DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler

ChunkKey = int | tuple[str, int]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "with",
}


class BM25Index:
    """
    BM25 Class Definition:
        BM25 is main algorithm for archiving every chunks created
        by the Chunker algorithm of the RAG. It will create multiple files
        for archiving chunks by their tokens count and will retrieve
        the right chunks using an inverted tokens index.
    """

    __STOPWORDS = STOPWORDS

    def __init__(
            self,
            k1: float = 1.2,
            b: float = 0.75,
            exact_match_boost: float = 1.15) -> None:
        """
        Init method of the BM25 algorithm
        """
        self.k1: float = k1
        self.b: float = b
        self.exact_match_boost: float = exact_match_boost
        self.inverted_index: dict[str, dict[ChunkKey, int]] = {}
        self.chunk_length: dict[ChunkKey, int] = {}
        self.chunk_count: int = 0
        self.total_chunk_length: float = 0.0
        self.average_chunk_length: float = 0.0
        self.token_stats: dict[str, int] = {}

    @classmethod
    def from_persisted(
            cls,
            inverted_index: dict[str, dict[ChunkKey, int]],
            token_stats: dict[str, int],
            chunk_length: dict[ChunkKey, int],
            chunk_count: int,
            k1: float = 1.2,
            b: float = 0.75,
            exact_match_boost: float = 1.15,
            ) -> "BM25Index":
        """Build a BM25 index from statistics stored in the database."""
        index = cls(k1, b, exact_match_boost)
        index.inverted_index = inverted_index
        index.chunk_length = chunk_length
        index.chunk_count = chunk_count
        index.total_chunk_length = float(sum(chunk_length.values()))
        index.average_chunk_length = (
            index.total_chunk_length / chunk_count if chunk_count else 0.0
        )
        index.token_stats = token_stats
        return index

    def create_index(
            self, chunks: List[Chunk]
            ) -> dict[str, dict[ChunkKey, int]]:
        """
        Main function of the BM25Index. This method will rank every chunks
        from a corpus and will rank them.
        """
        self.inverted_index.clear()
        self.chunk_length.clear()
        self.token_stats = {}
        for chunk in chunks:
            self.add_chunk(chunk)
        self.finalize()
        return self.inverted_index

    def __chunk_key(self, chunk: Chunk) -> ChunkKey:
        """Return a globally usable key for a chunk."""
        if chunk.file_path_hash:
            return (chunk.file_path_hash, chunk.id)
        return chunk.id

    def add_chunk(self, chunk: Chunk) -> None:
        """
        Register a Chunk into the BM25 inverted index

        This function will extract the tokens frequency of the chunk,
            the len of the chunks and register all this data into
            the chunk id key.
        """
        chunk_id = self.__chunk_key(chunk)
        tokens = chunk.tokens

        self.chunk_length[chunk_id] = len(tokens)
        for token in tokens:
            postings = self.inverted_index.setdefault(token, {})
            postings[chunk_id] = (postings.get(chunk_id, 0) + 1)

    def finalize(self) -> None:
        """
        Calculate the average chunk length
        """
        self.chunk_count = len(self.chunk_length)
        self.total_chunk_length = float(sum(self.chunk_length.values()))
        self.token_stats = {
            token: len(postings)
            for token, postings in self.inverted_index.items()
        }
        if self.chunk_count == 0:
            self.average_chunk_length = 0.0
            return
        self.average_chunk_length = (
            self.total_chunk_length / self.chunk_count
        )

    def score(self, query_tokens: List[str]) -> dict[ChunkKey, float]:
        """
        Calculate the final BM25 score for a given User query
        """
        query_tokens = list(dict.fromkeys(
            tokenize_text(" ".join(query_tokens))
        ))
        scores: dict[ChunkKey, float] = {}

        for token in query_tokens:
            postings = self.inverted_index.get(token)
            if not postings:
                continue
            idf = self.__idf(token)
            boost = 1.0 if token in self.__STOPWORDS \
                else self.exact_match_boost
            for chunk_id, term_frequency in postings.items():
                chunk_len = self.chunk_length[chunk_id]
                if self.average_chunk_length == 0.0:
                    continue
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * (
                        chunk_len / self.average_chunk_length
                    )
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                    idf * numerator / denominator * boost
                )

        return scores

    def search(
            self,
            query_tokens: List[str],
            top_k: int = 10) -> List[Tuple[ChunkKey, float]]:
        """
        Return the top_k candidate chunks for a given user query.
        """
        scores = self.score(query_tokens)
        return heapq.nlargest(
            top_k, scores.items(), key=lambda item: item[1]
        )

    def __idf(self, tokens: str) -> float:
        """
        Calculate the IDF (Inverse Document Frequency) of a token in all
            chunks.
        """
        df = self.token_stats.get(tokens)
        if df is None:
            postings = self.inverted_index.get(tokens)
            if not postings:
                return 0.0
            df = len(postings)
        if df == 0:
            return 0.0
        return math.log(
            1.0 + (self.chunk_count - df + 0.5) / (df + 0.5))


class DatabaseBM25Index:
    """BM25 matcher that reads only query postings from SQLite."""

    __STOPWORDS = STOPWORDS

    def __init__(
            self,
            database: "DataBaseHandler",
            k1: float = 1.2,
            b: float = 0.75,
            exact_match_boost: float = 1.15) -> None:
        """Initialize a database-backed BM25 matcher."""
        self.__database = database
        self.k1 = k1
        self.b = b
        self.exact_match_boost = exact_match_boost

    def search(
            self,
            query_tokens: List[str],
            top_k: int = 10) -> List[Tuple[ChunkKey, float]]:
        """Return ranked chunks for query tokens loaded from SQLite."""
        tokens = list(dict.fromkeys(
            tokenize_text(" ".join(query_tokens))
        ))
        rows, chunk_count, total_token_lengths = (
            self.__database.get_bm25_postings(tokens)
        )
        if not rows or chunk_count == 0:
            return []
        lengths = {
            (row[1], row[2]): row[5] for row in rows
        }
        average_length = total_token_lengths / chunk_count
        if average_length == 0.0:
            return []
        scores: dict[ChunkKey, float] = {}
        for token, hash_id, chunk_id, frequency, document_frequency, _ in rows:
            key = (hash_id, chunk_id)
            idf = math.log(
                1.0 + (chunk_count - document_frequency + 0.5) / (
                    document_frequency + 0.5
                )
            )
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * (
                    lengths[key] / average_length
                )
            )
            boost = 1.0 if token in self.__STOPWORDS else (
                self.exact_match_boost
            )
            scores[key] = scores.get(key, 0.0) + (
                idf * numerator / denominator * boost
            )
        return heapq.nlargest(
            top_k, scores.items(), key=lambda item: item[1]
        )
