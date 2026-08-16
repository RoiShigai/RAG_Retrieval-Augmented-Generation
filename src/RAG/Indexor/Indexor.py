from Chunker import PythonChunker, MarkDownChunker

DEFAULT_CHUNK_SIZE = 500


class Indexor:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        try:
            self.__chunker: dict = {
                ".py": PythonChunker(chunk_size),
                ".md": MarkDownChunker(chunk_size)
            }
        except (Exception) as e:
            print(f"Catch: {e}")
