from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import List


class ChunkType(Enum):
    PYTHON_FUNCTION = "python_function"
    PYTHON_CLASS = "python_class"
    PYTHON_DOCSTRING = "python_docstring"
    PYTHON_MODULE = "python_module"

    MARKDOWN_SECTION = "markdown section"
    MARKDOWN_CODE_BLOCK = "markdown_code_block"


@dataclass
class Chunk:
    """ DataClass that represent a chunk during the Indexing """
    file_path: Path
    start: int
    end: int
    chunk_type: ChunkType
    parent_id: int | None
    tokens: List[str]
