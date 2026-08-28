from ..Indexor import Chunk, MarkDownTokenizer
from collections import Counter
from typing import List, Tuple


class BM25Index:
    """
    BM25 Class Definition:
        BM25 is main algorithm for archiving every chunks created
        by the Chunker algorithm of the RAG. It will create multiple files
        for archiving chunks by their tokens count and will retrieve
        the right chunks using an inverted tokens index.
    """

    def __init__(self):
        """
        Init method of the BM25 algorithm
        """
        self.inverted_index: dict = dict()
        self.chunk_length: dict = dict()
        self.chunk_count: int = 0
        self.total_chunk_length: int = 0

    def add_chunk(self, chunk: Chunk) -> None:
        """
        Register a Chunk into the BM25 index

        This function will extract the tokens frequency of the chunk,
            the len of the chunks and register all this data into
            the chunk id key.
        """
        chunk_id = chunk.id
        tokens = chunk.tokens
        frequencies = Counter(tokens)

        self.chunk_length[chunk_id] = len(tokens)
        self.chunk_count += 1
        self.total_chunk_length += len(tokens)

        for token, frenquency in frequencies.items():
            if token not in self.inverted_index:
                self.inverted_index[token] = {}
            self.inverted_index[token][chunk_id] = frenquency

    def search(
            self,
            query: str,
            tokenizer: MarkDownTokenizer,
            top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Return the top_k candidate chunks for a given user query.
        """
        query_tokens = tokenizer.tokenize(query)
        candidates = set()

        for token in query_tokens:
            postings = self.inverted_index.get(token)
            if postings:
                candidates.update(postings.keys())

        scores = []

        for chunk_id in candidates:
            score = self._bm25_score(chunk_id, query_tokens)
            scores.append((chunk_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
