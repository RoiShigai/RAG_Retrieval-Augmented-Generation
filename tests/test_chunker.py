from src.Indexor.Chunker.FileChunker.PythonChunker import PythonChunker
from src.Indexor.Chunker.FileChunker.MarkDownChunker import MarkDownChunker
from src.Indexor.Chunker.Chunk import IdGenerator
from pathlib import Path

MAX_CHUNK_SIZE = 2000


def test_python_chunk_size_limit() -> None:
    chunker = PythonChunker(MAX_CHUNK_SIZE, IdGenerator())

    chunks = chunker.chunk(
        Path("src/Algorithm/Match.py")
    )
    assert all(
        chunk.end - chunk.start <= MAX_CHUNK_SIZE
        for chunk in chunks
    )


def test_python_chunk_limit_counts_source_characters(
        tmp_path: Path) -> None:
    source = "def f():\n    value = \"a very long string\"\n"
    path = tmp_path / "long.py"
    path.write_text(source)

    chunks = PythonChunker(10, IdGenerator()).chunk(path)

    assert len(chunks) > 1
    assert all(chunk.end - chunk.start <= 10 for chunk in chunks)


def test_markdown_chunk_limit_counts_source_characters(
        tmp_path: Path) -> None:
    source = "# Heading\nA paragraph with many words.\n"
    path = tmp_path / "long.md"
    path.write_text(source)

    chunks = MarkDownChunker(10, IdGenerator()).chunk(path)

    assert len(chunks) > 1
    assert all(chunk.end - chunk.start <= 10 for chunk in chunks)
