from ..Indexor.Chunker.Chunk import Chunk
from typing import List, Tuple
import math


class BM25Index:
    """
    BM25 Class Definition:
        BM25 is main algorithm for archiving every chunks created
        by the Chunker algorithm of the RAG. It will create multiple files
        for archiving chunks by their tokens count and will retrieve
        the right chunks using an inverted tokens index.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Init method of the BM25 algorithm
        """
        self.k1: float = k1
        self.b: float = b
        self.inverted_index: dict = dict()
        self.chunk_length: dict = dict()
        self.chunk_count: int = 0
        self.total_chunk_length: float = 0.0

    def create_index(self, chunks: List[Chunk]) -> dict:
        """
        Main function of the BM25Index. This method will rank every chunks
        from a corpus and will rank them.
        """
        for chunk in chunks:
            self.add_chunk(chunk)
        self.finalize()
        return self.inverted_index

    def add_chunk(self, chunk: Chunk) -> None:
        """
        Register a Chunk into the BM25 inverted index

        This function will extract the tokens frequency of the chunk,
            the len of the chunks and register all this data into
            the chunk id key.
        """
        chunk_id = chunk.id
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
        if self.chunk_count == 0:
            self.average_chunk_length = 0.0
            return
        self.average_chunk_length = (
            sum(self.chunk_length.values()) / self.chunk_count
        )

    def score(self, query_tokens: List[str]) -> dict[int, float]:
        """
        Calculate the final BM25 score for a given User query
        """
        query_tokens = list(dict.fromkeys(query_tokens))
        candidates = self.__get_candidates(query_tokens)
        scores: dict[int, float] = {}

        for chunk_id in candidates:
            total = 0.0
            for token in query_tokens:
                total += self.__score_token(token, chunk_id)
            scores[chunk_id] = total

        return scores

    def search(
            self,
            query_tokens: List[str],
            top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Return the top_k candidate chunks for a given user query.
        """
        scores = self.score(query_tokens)
        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True)[:top_k]

    def __idf(self, tokens: str) -> float:
        """
        Calculate the IDF (Inverse Document Frequency) of a given token in whole
            Chunks
        """
        postings = self.inverted_index.get(tokens)
        if not postings:
            return 0.0
        df = len(postings)
        return math.log(
            1.0 + (self.chunk_count - df + 0.5) / (df + 0.5))

    def __score_token(self, token: str, chunk_id: int) -> float:
        """
        Calculate the score for one token in one chunk
        """
        postings = self.inverted_index.get(token)

        if not postings:
            return 0.0
        tf = postings.get(chunk_id)
        if tf is None:
            return 0.0

        chunk_len = self.chunk_length[chunk_id]
        idf = self.__idf(token)

        numerator = tf * (self.k1 + 1.0)
        denominator = (
            tf + self.k1 * (
                1.0 - self.b + self.b * (
                    chunk_len / self.average_chunk_length
                )
            )
        )
        return idf * numerator / denominator

    def __get_candidates(self, query_tokens: List[str]) -> set[int]:
        """
        Return the candidate chunk id for a given User query
        """
        candidates: set[int] = set()

        for token in query_tokens:
            postings = self.inverted_index.get(token)
            if postings:
                candidates.update(postings.keys())
        return candidates
