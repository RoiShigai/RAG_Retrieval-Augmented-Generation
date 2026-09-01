import os
from pathlib import Path

from src.Algorithm.Match import BM25Index
from src.DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from src.Indexor.Indexor import Indexor


def test_reindex_keeps_tokens_for_unchanged_files(tmp_path: Path) -> None:
    database = DataBaseHandler(tmp_path / "index.db")
    first_file = tmp_path / "first.py"
    second_file = tmp_path / "second.py"
    first_file.write_text("def first():\n    return alpha\n")
    second_file.write_text("def second():\n    return beta\n")

    indexor = Indexor(database)
    initial_chunks = indexor.generate_chunks(tmp_path)
    database.store_bm25_index(BM25Index().create_index(initial_chunks))
    original_timestamp = second_file.stat().st_mtime

    first_file.write_text("def first():\n    return gamma\n")
    os.utime(first_file, (original_timestamp + 10, original_timestamp + 10))

    updated_chunks = Indexor(database).generate_chunks(tmp_path)
    tokens_by_file = {
        chunk.file_path.name: chunk.tokens
        for chunk in updated_chunks
    }

    assert any("gamma" in tokens for tokens in tokens_by_file.values())
    assert any("beta" in tokens for tokens in tokens_by_file.values())
    database.close()
