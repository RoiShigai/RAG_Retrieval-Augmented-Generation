from .FileChunker import FileChunker
from pathlib import Path


class MarkDownChunker(FileChunker):

    def chunk(self, path: Path) -> dict:
        """
        Chunk the file data so can be indexed
        """
