from abc import abstractmethod, ABC


class ChunkError(Exception):
    def __init__(self, msg: str) -> None:
        self.super(msg)


class Chunker(ABC):

    @abstractmethod
    def chunk_file(self, file_data: str | dict) -> dict:
        ...

    def __set_chunk_size(self, chunk_size: int) -> None:
        if chunk_size and chunk_size > 0 and chunk_size <= 2000:
            self.__chunk_size = chunk_size
        else:
            raise ChunkError(
                f"ChunkSizeError: Invalid Chunk size '{chunk_size}'" \
                f"Chunk size must be >0 and <=2000."
            )
