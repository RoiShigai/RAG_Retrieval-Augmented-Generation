from FileChunker import FileChunker
from pathlib import Path
from Chunk import Chunk
from typing import List
import ast


class PythonChunker(FileChunker):

    def chunk(self, path: Path, chunk_size: int) -> dict:
        """
        Chunk the file data so can be indexed
        """
        source: str = path.read_text()
        tree = ast.parse(source)
        chunks: List[Chunk] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunks.append(
                    self.__create_function_chunk(
                        path,
                        source,
                        node)
                )
                docstring = self.__get_docstring_node(node)
                if docstring:
                    chunks.append(
                        self.__create_docstring_chunk(
                            path,
                            source,
                            docstring
                        )
                    )
