from .FileChunker import FileChunker
from ..Chunk import Chunk, IdGenerator, ChunkType
from pathlib import Path
from typing import List
import re


class MarkDownChunker(FileChunker):

    def __init__(self, chunk_size: int, id_generator: IdGenerator) -> None:
        """
        Init method of the PythonFileCHunker class,
            This method create an instance of the PythonTokenizer
        """
        self.__id_generator = id_generator
        self.__max_chunk_size = chunk_size

    def chunk(self, path: Path) -> List[Chunk]:
        """
        Return a list of chunks from a MarkDown file
        """
        source = path.read_text(encoding="utf-8")
        sections = self.parse(source)
        chunks: List[Chunk] = []

        for section in sections:
            if section.parent is not None:
                continue
            chunks.extend(
                self.__chunk_section(
                    path, source, section
                )
            )
        return chunks

    def __chunk_section(
            self,
            path: Path,
            source: str,
            section: MarkdownSection) -> List[Chunk]:
        content = source[section.start:section.end]
        tokens = self.__tokenizer.tokenize(content)

        if len(tokens) <= self.__max_chunk_size:
            return [
                self.__create_chunk(
                    path,
                    section,
                    content,
                    tokens
                )
            ]

        return self.__split_section(path, source, section)

    def __split_section(
            self,
            path: Path,
            source: str,
            section: MarkdownSection) -> List[Chunk]:
        """
        Will split a section that is too big for the actual chunk size
        """
        chunks: List[Chunk] = []

        content_start = section.start
        for child in section.children:
            candidate = source[content_start:child.end]

            tokens = self.__tokenizer.tokenize(candidate)
            if len(tokens) <= self.__max_chunk_size:
                chunks.append(
                    self.__create_chunk(
                        path,
                        section,
                        candidate,
                        tokens,
                        start=content_start,
                        end=child.end
                    )
                )
                content_start = child.end
            else:
                chunks.extend(
                    self.__chunk_section(
                        path,
                        source,
                        child
                    )
                )
                content_start = child.end
        return chunks

    def __heading_path(self, section: MarkdownSection) -> List[str]:
        """
        Reconstruct the metadata path for child chunks.

        This function is used to keep the semantic between child chunk 
            so we can determine to which section the child chunk attach to
        """
        path: List[str] = []
        current: MarkdownSection | None = section

        while current is not None:
            path.append(current.title)
            current = current.parent
        path.reverse()

        return path
