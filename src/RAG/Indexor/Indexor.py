from Chunker import Chunker, Chunk
import Path

MAX_CHUNK_SIZE = 500


class Indexor:
    """
    Indexor Class Definition

    The Indexor is one of the mqin core element of this RAG.

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
        try:
            self.__chunker = Chunker(chunk_size)
        except (Exception) as e:
            print(f"Catch: {e}")

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
        for path in root_file.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in self.SUPPORTED_EXTENSION:
                continue
            chunks = self.chunker.chunk(path)
            for chunk in chunks:
                self._index_chunks(chunk)

    def __index_chunks(chunk: Chunk) -> None:
        print("[RAG LOG] - Indexing the chunks...")
        pass
