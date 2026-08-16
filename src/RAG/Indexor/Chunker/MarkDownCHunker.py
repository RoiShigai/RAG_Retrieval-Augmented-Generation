from .Chunker import Chunker


class MarkDownChunker(Chunker):

    def __init__(self, chunk_size: int = 2000) -> None:
        self.__set_chunk_size(chunk_size)

    def chunk_file(self, file_data: str | dict) -> dict:
        """
        Chunk the file data so can be indexed
        """
