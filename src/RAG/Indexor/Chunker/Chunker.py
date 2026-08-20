from .FileChunker import PythonChunker
from .Chunk import IdGenerator
from pathlib import Path


class ChunkError(Exception):
    def __init__(self, msg: str) -> None:
        self.super(msg)


class Chunker:

    def __init__(self, chunk_size: int) -> None:
        id_generator: IdGenerator = IdGenerator()
        self.tokenizer: dict = {
            ".py": PythonChunker(chunk_size, id_generator)
        }
        self.__max_chunk_size: int = chunk_size

    def chunk(self, path: Path) -> dict:
        """
        Return list of chunks from the given file.

        This function will automatically choose the right Tokenizer
            depending of the file suffix. Raise an Error if a file
            cannot be handle by the actual Tokenizer implementation.

        Parameters:
            path: The path object of the file to tokenize.

        Return:
            List[Chunk]: Return the list of chunks.
        """
        return self.tokenizer[path.suffix].chunk(
            path, self.__max_chunk_size
        )
