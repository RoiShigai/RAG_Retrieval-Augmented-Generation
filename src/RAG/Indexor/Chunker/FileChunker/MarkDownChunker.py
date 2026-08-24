from .FileChunker import FileChunker
from ..Chunk import Chunk, IdGenerator, ChunkType
from pathlib import Path


class MarkDownChunker(FileChunker):

    def __init__(self, chunk_size: int, id_generator: IdGenerator) -> None:
        """
        Init method of the PythonFileCHunker class,
            This method create an instance of the PythonTokenizer
        """
        self.__id_generator = id_generator
        self.__max_chunk_size = chunk_size

    def chunk(self, path: Path) -> dict:
        """
        Chunk the file data so can be indexed
        """
