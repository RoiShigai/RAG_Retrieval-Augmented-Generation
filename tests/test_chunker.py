from src.Indexor.Chunker.FileChunker.PythonChunker import PythonChunker
from src.Indexor.Chunker.FileChunker.MarkDownChunker import MarkDownChunker
from src.Indexor.Chunker.Chunk import ChunkType, IdGenerator
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


def test_python_chunks_include_context_and_filename_tokens(
        tmp_path: Path) -> None:
    path = tmp_path / "DataLoader.py"
    path.write_text(
        "class DataLoader:\n    def load_data(self):\n        return 1\n"
    )

    chunks = PythonChunker(2000, IdGenerator()).chunk(path)

    assert chunks
    assert all("python" in chunk.tokens for chunk in chunks)
    assert all("code" in chunk.tokens for chunk in chunks)
    assert all("data" in chunk.tokens for chunk in chunks)
    assert all("loader" in chunk.tokens for chunk in chunks)
    assert any("class" in chunk.tokens for chunk in chunks)


def test_long_token_is_searchable_in_bounded_chunks(tmp_path: Path) -> None:
    long_name = "a" * 30
    path = tmp_path / "long.py"
    path.write_text(f"def f():\n    value = {long_name}\n")

    chunks = PythonChunker(10, IdGenerator()).chunk(path)

    assert chunks
    assert all(chunk.end - chunk.start <= 10 for chunk in chunks)
    assert any(long_name in chunk.tokens for chunk in chunks)


def test_markdown_preamble_is_indexed(tmp_path: Path) -> None:
    path = tmp_path / "preamble.md"
    path.write_text("searchable preamble\n# Heading\nbody\n")

    chunks = MarkDownChunker(2000, IdGenerator()).chunk(path)

    assert any("searchable" in chunk.tokens for chunk in chunks)


def test_python_module_statements_are_indexed(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text(
        '"""module documentation"""\n'
        'MODULE_SETTING = "searchable setting"\n'
        "import pathlib\n"
    )

    chunks = PythonChunker(2000, IdGenerator()).chunk(path)

    assert chunks
    assert all(chunk.chunk_type == ChunkType.PYTHON_MODULE for chunk in chunks)
    assert any("searchable" in chunk.tokens for chunk in chunks)
    assert all("module" in chunk.tokens for chunk in chunks)


def test_split_function_chunks_keep_function_context(tmp_path: Path) -> None:
    path = tmp_path / "loader.py"
    path.write_text(
        "def load_model():\n"
        '    first = "a long value that must be split"\n'
        '    second = "another long value that must be split"\n'
    )

    chunks = PythonChunker(20, IdGenerator()).chunk(path)

    assert len(chunks) > 1
    assert all("load_model" in chunk.tokens for chunk in chunks)
    assert all("load" in chunk.tokens for chunk in chunks)
    assert all("model" in chunk.tokens for chunk in chunks)


def test_split_class_method_chunks_keep_class_and_method_context(
        tmp_path: Path) -> None:
    path = tmp_path / "runner.py"
    path.write_text(
        "class ModelRunner:\n"
        "    def initialize_cache(self):\n"
        '        value = "a long value that must be split"\n'
        '        other = "another long value that must be split"\n'
    )

    chunks = PythonChunker(25, IdGenerator()).chunk(path)

    assert len(chunks) > 1
    assert all("modelrunner" in chunk.tokens for chunk in chunks)
    assert all("initialize_cache" in chunk.tokens for chunk in chunks)
    assert all("initialize" in chunk.tokens for chunk in chunks)
    assert all("cache" in chunk.tokens for chunk in chunks)


def test_markdown_oversized_chunks_overlap(tmp_path: Path) -> None:
    path = tmp_path / "overlap.md"
    path.write_text(" ".join(f"word{index}" for index in range(80)))

    chunks = MarkDownChunker(40, IdGenerator()).chunk(path)

    assert len(chunks) > 1
    assert all(chunk.end - chunk.start <= 40 for chunk in chunks)
    assert any(
        current.start < previous.end
        for previous, current in zip(chunks, chunks[1:])
    )


def test_markdown_overlap_handles_minimum_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "small.md"
    path.write_text("abcdefghij")

    chunks = MarkDownChunker(1, IdGenerator()).chunk(path)

    assert chunks
    assert all(chunk.end - chunk.start <= 1 for chunk in chunks)
