import os
import sqlite3
from pathlib import Path

from src.DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from src.Indexor.Chunker.Chunk import Chunk, ChunkType


def test_chunk_and_bm25_posting_round_trip(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.md"
    source.write_text("# title\n", encoding="utf-8")
    assert database.check_file_modified(source)
    path_hash, content_hash = database.get_file_identity(source)
    database.update_file_metadata(source)
    chunk = Chunk(
        id=0,
        file_path=source,
        start=0,
        end=7,
        chunk_type=ChunkType.MARKDOWN_SECTION,
        parent_id=None,
        tokens=["title", "title"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    )
    database.replace_file_chunks(source, [chunk])
    database.store_bm25_index({
        "title": {(path_hash, 0): 2},
    })

    loaded = database.get_chunk(path_hash, 0)
    assert loaded.file_content_hash == content_hash
    assert loaded.tokens == ["title", "title"]
    assert database.get_bm25_postings(["title"]) == (
        [("title", path_hash, 0, 2, 1, 2)], 1, 2
    )
    database.close()


def test_bm25_posting_query_keeps_all_matching_postings(
        tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.md"
    source.write_text("# title\n", encoding="utf-8")
    path_hash, content_hash = database.get_file_identity(source)
    database.update_file_metadata(source)
    database.replace_file_chunks(source, [Chunk(
        id=0,
        file_path=source,
        start=0,
        end=8,
        chunk_type=ChunkType.MARKDOWN_SECTION,
        parent_id=None,
        tokens=["title"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    ), Chunk(
        id=1,
        file_path=source,
        start=0,
        end=8,
        chunk_type=ChunkType.MARKDOWN_SECTION,
        parent_id=None,
        tokens=["title"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    )])

    database.store_bm25_index({
        "title": {
            (path_hash, 0): 1,
            (path_hash, 1): 1,
        },
    })

    rows, total_chunks, total_lengths = database.get_bm25_postings(["title"])
    assert sorted(rows) == [
        ("title", path_hash, 0, 1, 2, 1),
        ("title", path_hash, 1, 1, 2, 1),
    ]
    assert (total_chunks, total_lengths) == (2, 2)
    database.close()


def test_timestamp_change_with_same_content_skips_chunking(
        tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.py"
    source.write_text("value = 1\n", encoding="utf-8")
    assert database.check_file_modified(source)
    database.update_file_metadata(source)
    original_time = source.stat().st_mtime

    os.utime(source, (original_time, original_time + 10))
    assert not database.check_file_modified(source)
    database.update_file_metadata(source)
    assert database.get_file_metadata(source)["modified_timestamp"] == (
        original_time + 10
    )
    database.close()


def test_changed_content_is_detected(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.py"
    source.write_text("value = 1\n", encoding="utf-8")
    assert database.check_file_modified(source)
    database.update_file_metadata(source)
    original_time = source.stat().st_mtime

    source.write_text("value = 2\n", encoding="utf-8")
    os.utime(source, (original_time, original_time + 10))
    assert database.check_file_modified(source)
    database.close()


def test_schema_contains_three_tables(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    connection = sqlite3.connect(tmp_path / "index.db")
    tables = {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert tables == {
        "files", "chunks", "chunk_tokens", "token_stats",
        "index_metadata",
    }
    connection.close()
    database.close()


def test_bm25_data_round_trip_and_statistics(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.md"
    source.write_text("# title\n", encoding="utf-8")
    path_hash, content_hash = database.get_file_identity(source)
    database.update_file_metadata(source)
    database.replace_file_chunks(source, [Chunk(
        id=0,
        file_path=source,
        start=0,
        end=8,
        chunk_type=ChunkType.MARKDOWN_SECTION,
        parent_id=None,
        tokens=["title", "title", "body"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    )])
    database.store_bm25_index({
        "title": {(path_hash, 0): 2},
        "body": {(path_hash, 0): 1},
    })

    rows, total_chunks, total_lengths = database.get_bm25_postings(
        ["title", "body"]
    )
    assert sorted(rows) == [
        ("body", path_hash, 0, 1, 1, 3),
        ("title", path_hash, 0, 2, 1, 3),
    ]
    assert (total_chunks, total_lengths) == (1, 3)
    database.close()


def test_refresh_bm25_metadata_preserves_postings(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.md"
    source.write_text("# title\n", encoding="utf-8")
    path_hash, content_hash = database.get_file_identity(source)
    database.update_file_metadata(source)
    database.replace_file_chunks(source, [Chunk(
        id=0,
        file_path=source,
        start=0,
        end=8,
        chunk_type=ChunkType.MARKDOWN_SECTION,
        parent_id=None,
        tokens=["title", "title", "body"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    )])

    database.refresh_bm25_metadata()

    assert database.get_bm25_postings(["title", "body"])[0] == [
        ("body", path_hash, 0, 1, 1, 3),
        ("title", path_hash, 0, 2, 1, 3),
    ]
    database.close()


def test_get_all_chunks_bulk_load_preserves_tokens(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.py"
    source.write_text("value = 1\n", encoding="utf-8")
    path_hash, content_hash = database.get_file_identity(source)
    database.update_file_metadata(source)
    database.replace_file_chunks(source, [Chunk(
        id=0,
        file_path=source,
        start=0,
        end=5,
        chunk_type=ChunkType.PYTHON_FUNCTION,
        parent_id=None,
        tokens=["value", "value", "one"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    ), Chunk(
        id=1,
        file_path=source,
        start=5,
        end=10,
        chunk_type=ChunkType.PYTHON_CLASS,
        parent_id=0,
        tokens=["second"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    )])

    chunks = database.get_all_chunks()

    assert [(chunk.id, chunk.tokens) for chunk in chunks] == [
        (0, ["one", "value", "value"]),
        (1, ["second"]),
    ]
    assert chunks[0].chunk_type is ChunkType.PYTHON_FUNCTION
    assert chunks[1].parent_id == 0
    database.close()


def test_synchronize_removes_deleted_file_and_postings(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    source = tmp_path / "sample.md"
    source.write_text("# title\n", encoding="utf-8")
    path_hash, content_hash = database.get_file_identity(source)
    database.update_file_metadata(source)
    database.replace_file_chunks(source, [Chunk(
        id=0,
        file_path=source,
        start=0,
        end=8,
        chunk_type=ChunkType.MARKDOWN_SECTION,
        parent_id=None,
        tokens=["title"],
        file_path_hash=path_hash,
        file_content_hash=content_hash,
    )])
    database.store_bm25_index({"title": {(path_hash, 0): 1}})
    source.unlink()

    assert database.synchronize_corpus(tmp_path)
    assert database.get_bm25_postings(["title"]) == ([], 0, 0)
    database.close()
