from FileChunker import FileChunker
from ..Tokenizer import PythonTokenizer
from pathlib import Path
from Chunk import Chunk, IdGenerator
from typing import List
import ast


class PythonChunker(FileChunker):
    """
    Chunker Class for the Python .py files

    This class will chunk a given python file into retrievable
        chunks, the method used to chunk python file is mean
        to store the semantic value of each method/function/class
        composing the file using the AST Python lib.
    A tiny chunk size can destroy the semantic value of the chunks,
        making the retriavable chances harder.
    """

    def __init__(self, chunk_size: int, id_generator: IdGenerator) -> None:
        """
        Init method of the PythonFileCHunker class,
            This method create an instance of the PythonTokenizer
        """
        self.__id_generator = id_generator
        self.__max_chunk_size = chunk_size
        self.__tokenizer = PythonTokenizer()

    def chunk(self, path: Path) -> List[Chunk]:
        """
        Chunking method for Python source file.

        This method will convert the file an python ast tree,
            check each node if it is a function Definition or a class.
            Then will convert each of this into retrieveable chunks.

        This method is mean to store the semantic and logic of python code
            using the Python ast lib, but with a tiny max_chunk value
            it will break all of the semantic.

        Parameters:
            path: Path | the path object of the corresponding python file.
            chunk_size: int | the max size of a chunk

        Return:
            List[Chunk]
        """
        source: str = path.read_text()
        tree = ast.parse(source)
        chunks: List[Chunk] = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                chunks.extend(
                    self.__chunk_function(
                        path,
                        source,
                        node)
                )
                docstring = node.get_docstring()
                if docstring:
                    chunks.append(
                        self.__create_docstring_chunk(
                            path,
                            source,
                            docstring
                        )

                    )
                continue
        return chunks

    def __chunk_function(
        self,
        path: Path,
        source: str,
        node: ast.FunctionDef) -> List[chunk]:
        """
        Transform each semantic node within the function into coherent chunks
        """
        offset: List[int] = self.__line_offset(source)
        start = self.__get_offset(offset, node.lineno, node.col_offset)
        end = self.__get_offset(offset, node.end_lineno, node.end_col_offset)
        function_source = source[start:end]
        tokens = self.__tokenizer.tokenize(function_source)
        return [
            Chunk(
                file_path=path,
                start=start,
                end=end,
                tokens=tokens
            )
        ]

    def chunk_docstring(
        self,
        path: Path,
        source: str,
        node: ast.Expr) -> List[Chunk]:
        """
        Return a List of Chunks containing the docstring of the associated function.
        """
        offset: List[int] = self.__line_offset
        start = self.__get_offset(offset, node.lineno, node.col_offset)
        end = self.__get_offset(
            offset,
            node.end_lineno,
            node.end_col_offset
        )
        docstring_source = source[start:end]
        tokens = self.__tokenizer.tokenize(docstring_source.strip('"""'))
        return [
            Chunk(
                file_path=path,
                start=start,
                end=end,
                tokens=tokens
            )
        ]

    def __line_offset(self, source: str) -> List[int]:
        """ List the offsets of each line """
        offset: List[int] = [0]

        for line in source.splitlines(keepends=True):
            offset.append(offset[-1] + len(line))
        return offset

    def __get_offset(self, offsets: List[int], line: int, column: int) -> int:
        """ Return the char position of the beginning statement """
        return offsets[line - 1] + column
