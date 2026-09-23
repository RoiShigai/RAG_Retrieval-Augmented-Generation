from pathlib import Path

import pytest

from Model import MinimalSearchResults, MinimalSource
from src.SLM.AnswerBatch import BatchAnswerParser, ContextBatchBuilder


def source(path: Path, start: int, end: int) -> MinimalSource:
    return MinimalSource(
        file_path=str(path),
        first_character_index=start,
        last_character_index=end,
    )


def test_context_builder_deduplicates_sources(tmp_path: Path) -> None:
    document = tmp_path / "document.md"
    document.write_text("shared context\nsecond context", encoding="utf-8")
    shared = source(document, 0, 14)
    other = source(document, 15, 29)
    results = [
        MinimalSearchResults(
            question_id="q1",
            question="first?",
            retrieved_sources=[shared, other],
        ),
        MinimalSearchResults(
            question_id="q2",
            question="second?",
            retrieved_sources=[shared],
        ),
    ]

    batches = list(ContextBatchBuilder().build_batches(results))

    assert len(batches) == 1
    assert [chunk.source_id for chunk in batches[0].chunks] == ["S1", "S2"]
    assert batches[0].questions[0].source_ids == ["S1", "S2"]
    assert batches[0].questions[1].source_ids == ["S1"]


def test_context_builder_splits_questions_at_character_limit(
        tmp_path: Path,
        ) -> None:
    document = tmp_path / "document.md"
    document.write_text("abcdefghijabcdefghij", encoding="utf-8")
    results = [
        MinimalSearchResults(
            question_id=f"q{index}",
            question=f"question {index}",
            retrieved_sources=[source(document, index * 10, (index + 1) * 10)],
        ) for index in range(2)
    ]
    one_source_length = len(
        f"[S1]\nFile: {document}\nabcdefghij"
    )

    batches = list(ContextBatchBuilder(
        max_char_length=one_source_length * 2 - 1
    ).build_batches(results))

    assert len(batches) == 2
    assert [batch.questions[0].question_id for batch in batches] == [
        "q0", "q1"
    ]
    assert all(len(batch.chunks) == 1 for batch in batches)


def test_context_builder_keeps_sources_for_each_question(
        tmp_path: Path,
        ) -> None:
    document = tmp_path / "document.md"
    document.write_text("first source\nsecond source", encoding="utf-8")
    results = [
        MinimalSearchResults(
            question_id="q1", question="question", retrieved_sources=[
                source(document, 0, 12), source(document, 13, 26)
            ]
        )
    ]

    batch = list(ContextBatchBuilder().build_batches(results))[0]

    assert [chunk.text for chunk in batch.chunks] == [
        "first source\n", "second source"
    ]
    assert batch.questions[0].source_ids == ["S1", "S2"]


def test_batch_parser_requires_all_question_ids() -> None:
    result = MinimalSearchResults(
        question_id="q1", question="question", retrieved_sources=[]
    )
    batch = list(ContextBatchBuilder().build_batches([result]))[0]

    parsed = BatchAnswerParser().parse(
        '[{"question_id":"q1","answer":"answer"}]', batch
    )

    assert parsed[0].question_id == "q1"
    assert parsed[0].answer == "answer"


def test_batch_parser_rejects_missing_question() -> None:
    result = MinimalSearchResults(
        question_id="q1", question="question", retrieved_sources=[]
    )
    batch = list(ContextBatchBuilder().build_batches([result]))[0]

    with pytest.raises(ValueError):
        BatchAnswerParser().parse("[]", batch)
