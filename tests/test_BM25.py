from src.Algorithm.Match import BM25Index
from src.Indexor.Chunker.Chunk import Chunk, ChunkType

chunk_01 = Chunk(
    id=1,
    file_path="src/Indexor",
    start=1,
    end=23,
    chunk_type=ChunkType.PYTHON_CLASS,
    parent_id=None,
    tokens=["cuda", "compiler", "cuda"]
)

chunk_02 = Chunk(
    id=2,
    file_path="src/Compiler",
    start=56,
    end=63,
    chunk_type=ChunkType.PYTHON_CLASS,
    parent_id=None,
    tokens=["python", "compiler", "python"]
)


chunk_03 = Chunk(
    id=3,
    file_path="src/debug",
    start=128,
    end=146,
    chunk_type=ChunkType.PYTHON_CLASS,
    parent_id=None,
    tokens=["python", "compiler", "debug", "linux"]
)


def test_add_chunk01():
    index = BM25Index()
    index.add_chunk(chunk_01)

    assert index.chunk_length[1] == 3
    assert index.inverted_index["cuda"][1] == 2
    assert index.inverted_index["compiler"][1] == 1


def test_add_chunk02():
    index = BM25Index()
    index.add_chunk(chunk_01)
    index.add_chunk(chunk_02)

    assert index.chunk_length[1] == 3
    assert index.inverted_index["cuda"][1] == 2
    assert index.inverted_index["compiler"][1] == 1

    assert index.chunk_length[2] == 3
    assert index.inverted_index["python"][2] == 2
    assert index.inverted_index["compiler"][1] == 1


def test_chunk_length():
    index = BM25Index()
    index.add_chunk(chunk_01)
    index.add_chunk(chunk_02)
    index.add_chunk(chunk_03)

    index.finalize()

    assert index.chunk_count == 3
    assert round(index.average_chunk_length, 2) == 3.33


def test_ranking():
    index = BM25Index()
    index.add_chunk(chunk_01)
    index.add_chunk(chunk_02)
    index.add_chunk(chunk_03)
    index.finalize()

    results = index.search(
        ["compiler", "python"], top_k=3
    )
    assert results[0][0] == 2


def test_unknown_query():
    index = BM25Index()
    index.add_chunk(chunk_01)
    index.add_chunk(chunk_02)
    index.add_chunk(chunk_03)
    index.finalize()

    results = index.search(
        ["quantum"], top_k=3
    )
    assert results == []


def test_basic_k():
    index = BM25Index()
    index.add_chunk(chunk_01)
    index.add_chunk(chunk_02)
    index.add_chunk(chunk_03)
    index.finalize()

    results = index.search(
        ["compiler", "python"], top_k=2
    )

    assert len(results) == 2


def test_persisted_index_has_same_ranking() -> None:
    original = BM25Index()
    original.create_index([chunk_01, chunk_02, chunk_03])
    restored = BM25Index.from_persisted(
        original.inverted_index,
        original.token_stats,
        original.chunk_length,
        original.chunk_count,
    )

    assert restored.search(["compiler", "python"], top_k=3) == (
        original.search(["compiler", "python"], top_k=3)
    )
