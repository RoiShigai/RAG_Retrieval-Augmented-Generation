from FileChunker import FileChunker
import Path


class MarkDownChunker(FileChunker):

    def __init__(self, chunk_size: int = 2000) -> None:
        self.__set_chunk_size(chunk_size)

    def chunk(self, path: Path, max_size: int) -> dict:
        """
        Chunk the file data so can be indexed
        """
