from FileChunker import FileChunker
from ..Tokenizer import PythonTokenizer
from pathlib import Path
from Chunk import Chunk, IdGenerator
from typing import List, Tuple
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
        offset = self.__line_offset(source)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunks.extend(
                    self.__chunk_class(
                        path, source, node, offset, None
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(
                    self.__chunk_function(
                        path, source, node, offset, None
                    )
                )
        return chunks

    def __chunk_function(
        self,
        path: Path,
        source: str,
        node: ast.FunctionDef,
        offset: List[int],
        parent_id: int | None = None) -> List[chunk]:
        """
        Take the input function node and return a list of chunk representing it
        """
        start, end = self.__node_range(offset, node)
        tokens = self.__tokenizer.tokenize(source[start:end])

        if len(tokens) <= self.__max_chunk_size:
            return [
                Chunk(
                    id=self.__id_generator.next(),
                    path=path,
                    start=start,
                    end=end,
                    tokens=tokens,
                    kind="function",
                    name=node.name,
                    parent_id=parent_id
                )
            ]
        return self.__split_function(path, source, offset, node, parent_id)

    def __split_function(
            self,
            path: Path,
            source: str,
            offset: List[int],
            node: ast.FunctionDef | ast.AsyncFunctionDef,
            parent_id: int | None = None) -> List[Chunk]:
        """ Try to split the function node by statement into multiple chunks """
        chunks: List[Chunk] = []

        if not node.body:
            start, end = self.__node_range(offset, node)
            return [
                Chunk(
                    id=self.__id_generator.next(),
                    path=path,
                    start=start,
                    end=end,
                    tokens=self.__tokenizer.tokenize(source[start:end]),
                    kind="function",
                    name=node.name,
                    parent_id=parent_id
                )
            ]
        current_start = None
        current_end = None
        for statement in node.body:
            statement_start, statement_end = self.__node_range(
                offset,
                statement
            )
            if current_start is None:
                current_start = statement_start
                current_end = statement_end
                continue
            candidate = source[current_start: statement_end]
            candidate_tokens = self.__tokenizer.tokenize(candidate)

            if len(candidate_tokens) < self.__max_chunk_size:
                current_end = statement_end
                continue

            chunks.append(
                Chunk(
                    id=self.__id_generator.next(),
                    path=path,
                    start=current_start,
                    end=current_end,
                    tokens=self.__tokenizer.tokenize(
                        source[current_start:current_end]
                    ),
                    kind="function",
                    name=node.name,
                    parent_id=parent_id
                )
            )
            current_start = statement_start
            current_end = statement_end

        if current_start is not None and current_end is not None:
            chunks.append(
                Chunk(
                    id=self.__id_generator.next(),
                    path=path,
                    start=current_start,
                    end=current_end,
                    tokens=self.__tokenizer.tokenize(
                        source[current_start:current_end]
                    ),
                    kind="function",
                    name=node.name,
                    parent_id=parent_id
                )
            )
        return chunks

    def __chunk_class(
            self,
            path: Path,
            source: str,
            offset: List[int],
            node: ast.ClassDef,
            parent_id: int | None = None) -> List[Chunk]:
        start, end = self.__node_range(offset, node)
        tokens = self.__tokenizer.tokenize(source[start:end])

        if len(tokens) <= self.__max_chunk_size:
            return [
                Chunk(
                    id=self.__id_generator.next(),
                    path=path,
                    start=start,
                    end=end,
                    tokens=self.__tokenizer.tokenize(source[start:end]),
                    kind="class",
                    name=node.name,
                    parent_id=parent_id
                )
            ]
        return self.__split_class(path, source, offset, node, parent_id)

    def __split_class(
            self,
            path: Path,
            source: str,
            offset: List[int],
            node: ast.ClassDef,
            parent_id: int | None = None) -> List[Chunk]:
        """ Return a List of Chunk representing the class """
        chunks: List[Chunk] = []
        current_start: int | None = None
        current_end: int | None = None

        for child in node.body:
            child_start, child_end = self.__node_range(
                offset,
                child
            )
            if current_start is None:
                current_start = child_start
                current_end = child_end
                continue
            candidate = source[current_start:child_end]
            candidate_tokens = self.__tokenizer.tokenize(candidate)

            if len(candidate_tokens) <= self.__max_chunk_size:
                current_end = child_end
                continue

            chunks.append(
                Chunk(
                    id=self.__id_generator.next(),
                    path=path,
                    start=current_start,
                    end=current_end,
                    tokens=self.__tokenizer.tokenize(
                        source[current_start:current_end]
                    ),
                    kind="class_part",
                    name=node.name,
                    parent_id=parent_id
                )
            )
            current_start = child_start
            current_end = child_end

        if current_start is not None and current_end is not None:
            chunks.append(
                Chunk(
                    id=self.__id_generator.next(),
                    path=path,
                    start=current_start,
                    end=current_end,
                    tokens=self.__tokenizer.tokenize(
                        source[current_start:current_end]
                    ),
                    kind="class_part",
                    name=node.name,
                    parent_id=parent_id
                )
            )
        return chunks

    def __statement_children(
        self,
        node: ast.AST,
    ) -> list[ast.stmt]:
        """
        Utils to check what type of node is evaluated
            during recursive chunking for tiny max_chunk
        """
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            return list(node.body)
        if isinstance(node, ast.If):
            return [
                *node.body,
                *node.orelse,
            ]
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return [
                *node.body,
                *node.orelse,
            ]
        if isinstance(node, (ast.While,)):
            return [
                *node.body,
                *node.orelse,
            ]
        if isinstance(node, ast.With):
            return list(node.body)
        if isinstance(node, ast.AsyncWith):
            return list(node.body)
        if isinstance(node, ast.Try):
            return [
                *node.body,
                *[
                    stmt
                    for handler in node.handlers
                    for stmt in handler.body
                ],
                *node.orelse,
                *node.finalbody,
            ]
        return []

    def __line_offset(self, source: str) -> List[int]:
        """ List the offsets of each line """
        offset: List[int] = [0]

        for line in source.splitlines(keepends=True):
            offset.append(offset[-1] + len(line))
        return offset

    def __get_offset(self, offsets: List[int], line: int, column: int) -> int:
        """ Return the char position of the beginning statement """
        return offsets[line - 1] + column

    def __node_range(self, node: ast.AST, offset: List[int]) -> Tuple[int, int]:
        """ Return the start and end char position of the node """
        start = self.__get_offset(offset, node.lineno, node.col_offset)
        end = self.__get_offset(offset, node.end_lineno, node.end_col_offset)

        return start, end

    def __fit(self, source: str, start: int, end: int) -> bool:
        """ Check if the token inside the node is not above max_chunk_size """
        text = source[start:end]
        tokens = self.__tokenizer.tokenize(text)

        return len(tokens) <= self.__max_chunk_size
