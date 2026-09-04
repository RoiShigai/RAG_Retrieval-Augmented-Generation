from pathlib import Path

from src.Model.MinimalSource import MinimalSource
from src.RAG import RAG


def test_search_returns_minimal_sources(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "sample.md"
    source.write_text("# title\nsearchable text\n", encoding="utf-8")

    rag = RAG(tmp_path / "index.db", corpus)
    rag.index()
    results = rag.search("searchable", 1)

    assert len(results) == 1
    assert isinstance(results[0], MinimalSource)
    assert results[0].file_path == str(source.resolve())
    assert results[0].first_character_index == 0
    assert results[0].last_character_index == len(
        source.read_text(encoding="utf-8")
    )
