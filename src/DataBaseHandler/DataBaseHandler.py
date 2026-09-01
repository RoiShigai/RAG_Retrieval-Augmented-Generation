from pathlib import Path
from Indexor.Chunker.Chunk import Chunk


class DataBaseHandler:
    """
    DataBaseHandler Class Definition

    The StoringManager is the class used to check the database
        and will perform every interaction with it.
    """

    def __init__(self, database: Path) -> None:
        """
            StoringManager Init Method

            Parameters:
                database: path to the database it will interact with
        """
        self.__database: Path = database

    def get_file_metadata(self, filename: Path) -> dict:
        """
            Return the file metadata stored in the database.
        """

    def check_file_modified(self, filename: Path) -> bool:
        """
            Check if the file has been modified since the last corpus indexing.
        """

    def update_file_metadata(self, filename: Path) -> None:
        """
            Store or Update the current file metadata into the database.
        """

    def store_chunks(self, chunk: Chunk) -> None:
        """
            Store the given chunk into the database.
        """

    def get_chunk(self, chunk_id: str) -> Chunk:
        """
            Return the corresponding chunk for a given id
        """

    def store_bm25_index(self, index: dict) -> None:
        """
            Store the Index created by the BM25 algorithm
        """

    def load_bm25_index(self) -> None:
        """
            Load the BM25 index from the databasa
        """
