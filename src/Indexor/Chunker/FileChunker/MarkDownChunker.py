from ..Tokenizer.MarkDownTokenizer import MarkDownTokenizer
from ..Tokenizer.MarkDownTokenizer import MarkdownSection
from ..Chunk import Chunk, IdGenerator, ChunkType
from .FileChunker import FileChunker
from pathlib import Path
from typing import List


class MarkDownChunker(FileChunker):

    def __init__(self, chunk_size: int, id_generator: IdGenerator) -> None:
        """
        Init method of the PythonFileCHunker class,
            This method create an instance of the PythonTokenizer
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        self.__id_generator = id_generator
        self.__max_chunk_size = chunk_size
        self.__tokenizer = MarkDownTokenizer()

    def chunk(self, path: Path) -> List[Chunk]:
        """
        Return a list of chunks from a MarkDown file
        """
        source = path.read_text(encoding="utf-8")
        sections = self.__tokenizer.parse(source)
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
        """
        Convert a MarkDownSection object into a Chunk
        """
        content = source[section.start:section.end]
        tokens = self.__tokenizer.tokenize(content)

        if len(content) <= self.__max_chunk_size:
            return [
                self.__create_chunk(
                    path,
                    section,
                    content,
                    ChunkType.MARKDOWN_SECTION,
                    tokens,
                    None,
                    section.start,
                    section.end
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

        content_start: int = section.start
        for child in section.children:
            candidate: str = source[content_start:child.start]

            tokens = self.__tokenizer.tokenize(candidate)
            if len(candidate) <= self.__max_chunk_size:
                chunks.append(
                    self.__create_chunk(
                        path,
                        section,
                        candidate,
                        ChunkType.MARKDOWN_SECTION,
                        tokens,
                        None,
                        content_start,
                        child.start
                    )
                )
                content_start = child.start
            else:
                chunks.extend(
                    self.__chunk_section(
                        path,
                        source,
                        child
                    )
                )
                content_start = child.end
        if content_start < section.end:
            chunks.extend(self.__source_chunks(
                path, source, content_start, section.end, section, None
            ))
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

    def __create_chunk(
            self,
            path: Path,
            section: MarkdownSection,
            source: str,
            chunk_type: ChunkType,
            tokens: List[str],
            parent_id: int | None,
            start: int,
            end: int) -> Chunk:
        """
        Return a chunk with the actual data
        """
        return Chunk(
            id=self.__id_generator.next(),
            file_path=path,
            start=start,
            end=end,
            chunk_type=chunk_type,
            parent_id=parent_id,
            tokens=tokens
            )

    def __source_chunks(
            self,
            path: Path,
            source: str,
            start: int,
            end: int,
            section: MarkdownSection,
            parent_id: int | None) -> List[Chunk]:
        """Split an oversized Markdown leaf on lines and characters."""
        chunks: List[Chunk] = []
        current = start
        for line in source[start:end].splitlines(keepends=True):
            line_end = current + len(line)
            for chunk_start in range(current, line_end, self.__max_chunk_size):
                chunk_end = min(chunk_start + self.__max_chunk_size, line_end)
                chunks.append(self.__create_chunk(
                    path,
                    section,
                    source[chunk_start:chunk_end],
                    ChunkType.MARKDOWN_SECTION,
                    self.__tokenizer.tokenize(source[chunk_start:chunk_end]),
                    parent_id,
                    chunk_start,
                    chunk_end
                ))
            current = line_end
        return chunks
