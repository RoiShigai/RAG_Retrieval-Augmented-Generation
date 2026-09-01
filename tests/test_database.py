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
    assert loaded.tokens == ["title"]
    assert database.load_bm25_index() == {"title": {(path_hash, 0): 2}}
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
    assert tables == {"files", "chunks", "reverse_key"}
    connection.close()
    database.close()
