from .Chunker.FileChunker.MarkDownChunker import MarkDownChunker
from .Chunker.FileChunker.PythonChunker import PythonChunker
from .Chunker.Chunk import IdGenerator, Chunk
from typing import List
from pathlib import Path

MAX_CHUNK_SIZE = 500


class Indexor:
    """
    Indexor Class Definition

    The Indexor is one of the main core element of this RAG.

    It will gather all the supported files starting from a given root folder
        and Chunks them accordingly to the max chunk size given
        by the user. If no chunk size has been define it will
        be set at MAX_CHUNK_SIZE by default.

    After the chunk phase, all of the chunk will get ranked using
        the BM25 Algorithm and the Indexor will store all of the data
        using reverse key into "project/data/processed"
    """

    SUPPORTED_EXTENSION: set = {".py", ".md"}

    def __init__(self, chunk_size: int = MAX_CHUNK_SIZE) -> None:
#       if chunk_size > MAX_CHUNK_SIZE:
#           raise RagError(
#               f"Chunk size cannot go above {MAX_CHUNK_SIZE}"
#           )
        self.__id_generator = IdGenerator()
        self.__chunker = {
            ".py": PythonChunker(chunk_size, self.__id_generator),
            ".md": MarkDownChunker(chunk_size, self.__id_generator)
        }

    def index(self, root_file: Path) -> None:
        """
        Main function for Indexing files into chunks.
        This function will start to chunkenize every supported file
            into chunks containing the appropriate metadata,
            and then index it.

        Parameter:
            root_file: the directory where the Indexer will gather the files.

        Return:
            This function return None but will display logs
                about the indexing process
        """
        chunks: List[Chunk] = []
        for path in root_file.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in self.SUPPORTED_EXTENSION:
                continue
            print(f"[DEBUG]: current file {path}")
            chunks = self.__chunker[path.suffix].chunk(path)
            print(f"chunks type: {type(chunks)}")
            if chunks:
                for chunk in chunks:
                    chunk.debug_chunk()
                    print(f"token: {chunk.tokens}")
                self.__index_chunks(chunks)

    def __index_chunks(self, chunks: List[Chunk]) -> None:
        print("[RAG LOG] - Indexing the chunks...")
        pass


if __name__ == "__main__":
    indexor = Indexor(700)
    indexor.index(Path("vllm-0.10.1"))
