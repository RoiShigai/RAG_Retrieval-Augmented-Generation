import os
import sqlite3
from pathlib import Path

from src.DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from src.Indexor.Chunker.Chunk import Chunk, ChunkType


def test_chunk_and_reverse_key_round_trip(tmp_path: Path) -> None:
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
    assert database.load_bm25_index() == {"title": {(path_hash, 0): 2}}
    database.close()


def test_load_bm25_index_keeps_all_postings(tmp_path: Path) -> None:
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

    assert database.load_bm25_index() == {
        "title": {
            (path_hash, 0): 1,
            (path_hash, 1): 1,
        },
    }
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
        "files", "chunks", "reverse_key", "chunk_tokens", "token_stats",
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

    data = database.load_bm25_data()
    assert data[1] == {"title": 1, "body": 1}
    assert data[2] == {(path_hash, 0): 3}
    assert data[3] == 1
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
    assert database.load_bm25_index() == {}
    assert database.load_bm25_data()[1:] == ({}, {}, 0)
    database.close()
