from .Chunker.FileChunker.MarkDownChunker import MarkDownChunker
from .Chunker.FileChunker.PythonChunker import PythonChunker
from .Chunker.Chunk import IdGenerator, Chunk
from ..DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
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

    SUPPORTED_EXTENSION: set[str] = {".py", ".md"}

    def __init__(
            self,
            database: DataBaseHandler,
            chunk_size: int = MAX_CHUNK_SIZE) -> None:
        self.__database = database
        self.__id_generator = IdGenerator()
        self.__chunker = {
            ".py": PythonChunker(chunk_size, self.__id_generator),
            ".md": MarkDownChunker(chunk_size, self.__id_generator)
        }

    def generate_chunks(self, root_file: Path) -> List[Chunk]:
        """
        Main function for Indexing files into chunks.
        This function will start to chunkenize every supported file
            into chunks containing the appropriate metadata,
            and return a chunk list of the corresponding root file.

        Parameter:
            root_file: the directory where the Indexer will gather the files.

        Return:
            This function will return a list of chunks
        """
        fresh_chunks: dict[tuple[str, int], Chunk] = {}

        for path in root_file.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in self.SUPPORTED_EXTENSION:
                continue

            print(f"[DEBUG]: current file {path}")
            modified = self.__database.check_file_modified(path)
            metadata = self.__database.get_file_metadata(path)
            if modified:
                print(f"Chunking {path}")
                path_hash, content_hash = (
                    self.__database.get_file_identity(path)
                )
                self.__id_generator.reset()
                chunks = self.__chunker[path.suffix].chunk(path)
                for chunk in chunks:
                    chunk.file_path_hash = path_hash
                    chunk.file_content_hash = content_hash
                    fresh_chunks[(path_hash, chunk.id)] = chunk
                self.__database.update_file_metadata(path)
                self.__database.replace_file_chunks(path, chunks)
            elif metadata is not None and path.stat().st_mtime != metadata[
                    "modified_timestamp"]:
                self.__database.update_file_metadata(path)

        chunks_by_key = {
            (chunk.file_path_hash, chunk.id): chunk
            for chunk in self.__database.get_all_chunks()
        }
        chunks_by_key.update(fresh_chunks)
        return list(chunks_by_key.values())
