from abc import ABC, abstractmethod
from ..Chunk import Chunk
from pathlib import Path
from typing import List


class FileChunker(ABC):

    @abstractmethod
    def chunk(self, path: Path, chunk_size: int) -> List[Chunk]:
        pass
