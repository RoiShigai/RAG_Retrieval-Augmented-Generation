from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import List


class ChunkType(Enum):
    PYTHON_FUNCTION = "function"
    PYTHON_CLASS = "class"
    PYTHON_CLASS_PART = "class_part"
    PYTHON_DOCSTRING = "docstring"
    PYTHON_MODULE = "module"

    MARKDOWN_SECTION = "markdown_section"
    MARKDOWN_CODE_BLOCK = "markdown_code_block"


class IdGenerator:
    """
    IdGenerator Class
    Used to generate the ID for each Chunk
    """

    def __init__(self) -> None:

        self.__next_id: int = 0

    def next(self) -> int:
        """
        Return the current number hold by the generator
        Each time method is called, the number hold
            by the generator is incremented by 1.
        """
        chunk_id = self.__next_id
        self.__next_id += 1
        return chunk_id


@dataclass
class Chunk:
    """ DataClass that represent a chunk during the Indexing """
    id: int
    file_path: Path
    start: int
    end: int
    chunk_type: ChunkType
    parent_id: int | None
    tokens: List[str]

    def debug_chunk(self) -> None:
        print(f"{self.id}, {self.file_path}, {self.start}, {self.end}, {self.chunk_type}, {self.parent_id}")
