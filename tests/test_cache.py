from pathlib import Path

from src.DataHandler.CacheHandler.CacheHandler import CacheHandler
from src.DataHandler.DatabaseHandler.DataBaseHandler import DataBaseHandler
from src.Model import MinimalAnswer, MinimalSource


def test_cache_round_trip_and_version_validation(tmp_path: Path) -> None:
    cache = CacheHandler(tmp_path / "cache")
    source = MinimalSource(
        file_path="/tmp/example.py",
        first_character_index=0,
        last_character_index=12,
    )
    answer = MinimalAnswer(
        question_id="question",
        question="question",
        retrieved_sources=[source],
        answer="cached answer",
    )

    cache.store_search_result("question", 2, "2:1", [source])
    cache.store_answer("question", 2, "2:1", answer)

    assert cache.get_search_result("question", 2, "2:1")[0].model_dump() == (
        source.model_dump()
    )
    assert cache.get_answer("question", 2, "2:1").model_dump() == (
        answer.model_dump()
    )
    assert cache.get_search_result("question", 2, "2:2") is None
    assert cache.get_answer("question", 2, "2:2") is None
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_database_version_is_persistent_and_changes_with_index_data(
        tmp_path: Path) -> None:
    database_path = tmp_path / "index.db"
    source = tmp_path / "sample.md"
    source.write_text("# title\n", encoding="utf-8")
    database = DataBaseHandler(database_path)

    initial_version = database.get_database_version()
    assert initial_version.startswith("2:")
    path_hash, content_hash = database.get_file_identity(source)
    database.update_file_metadata(source)
    database.close()

    reopened = DataBaseHandler(database_path)
    assert reopened.get_database_version() == initial_version
    reopened.replace_file_chunks(source, [])
    assert reopened.get_database_version() != initial_version
    assert path_hash
    assert content_hash
    reopened.close()
