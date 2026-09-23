import json
from collections.abc import Iterator, Sequence

from pydantic import BaseModel, ConfigDict

from Model import MinimalSearchResults, MinimalSource
from Helper import retrieve_text_from_source


class ContextChunk(BaseModel):
    """Represent one deduplicated source chunk in an answer batch."""

    source_id: str
    source: MinimalSource
    text: str


class BatchQuestion(BaseModel):
    """Represent one question and its allowed context chunks."""

    question_id: str
    question: str
    source_ids: list[str]


class AnswerBatch(BaseModel):
    """Contain questions and shared context for one model request."""

    chunks: list[ContextChunk]
    questions: list[BatchQuestion]


class GeneratedBatchAnswer(BaseModel):
    """Represent one validated answer returned by the language model."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    answer: str


class ContextBatchBuilder:
    """Build bounded batches with deduplicated retrieved source chunks."""

    def __init__(
            self,
            max_questions: int = 5,
            max_char_length: int = 15000,
            ) -> None:
        """Initialize the builder with question and context limits."""
        if max_questions < 1 or max_char_length < 1:
            raise ValueError("Batch limits must be positive")
        self.__max_questions = max_questions
        self.__max_char_length = max_char_length

    def build_batches(
            self,
            search_results: Sequence[MinimalSearchResults],
            ) -> Iterator[AnswerBatch]:
        """Yield bounded answer batches in the input order."""
        current: list[MinimalSearchResults] = []
        for result in search_results:
            candidate = [*current, result]
            too_many_questions = len(candidate) > self.__max_questions
            too_many_chars = (
                self.__context_length(candidate) > self.__max_char_length
            )
            if current and (too_many_questions or too_many_chars):
                yield self.__build_batch(current)
                current = []
            current.append(result)
        if current:
            yield self.__build_batch(current)

    def __build_batch(
            self,
            search_results: Sequence[MinimalSearchResults],
            ) -> AnswerBatch:
        source_data: dict[tuple[str, int, int], ContextChunk] = {}
        question_sources: dict[str, list[str]] = {}

        for result in search_results:
            question_sources[result.question_id] = []
            for source in result.retrieved_sources:
                key = self.__source_key(source)
                if key not in source_data:
                    source_id = f"S{len(source_data) + 1}"
                    text = retrieve_text_from_source(source)
                    source_data[key] = ContextChunk(
                        source_id=source_id,
                        source=source,
                        text=text,
                    )
                question_sources[result.question_id].append(
                    source_data[key].source_id
                )

        selected = list(source_data.values())
        selected_ids = {chunk.source_id for chunk in selected}

        questions = [
            BatchQuestion(
                question_id=result.question_id,
                question=result.question,
                source_ids=[
                    source_id
                    for source_id in question_sources[result.question_id]
                    if source_id in selected_ids
                ],
            )
            for result in search_results
        ]
        return AnswerBatch(chunks=selected, questions=questions)

    def __context_length(
            self,
            search_results: Sequence[MinimalSearchResults],
            ) -> int:
        """Return the rendered source context length for these questions."""
        source_data: dict[tuple[str, int, int], str] = {}
        for result in search_results:
            for source in result.retrieved_sources:
                key = self.__source_key(source)
                if key not in source_data:
                    source_data[key] = retrieve_text_from_source(source)
        return len("\n\n".join(
            f"[S{index}]\nFile: {key[0]}\n{text}"
            for index, (key, text) in enumerate(source_data.items(), 1)
        ))

    @staticmethod
    def __source_key(source: MinimalSource) -> tuple[str, int, int]:
        """Return the stable identity of a retrieved source chunk."""
        return (
            source.file_path,
            source.first_character_index,
            source.last_character_index,
        )


class BatchAnswerParser:
    """Parse and validate structured model answers for one batch."""

    def parse(
            self,
            response: str,
            batch: AnswerBatch,
            ) -> list[GeneratedBatchAnswer]:
        """Return ordered answers or raise ``ValueError`` on invalid output."""
        payload = self.__decode_json(response)
        if not isinstance(payload, list):
            raise ValueError("Batch response must be a JSON array")

        answers = [
            GeneratedBatchAnswer.model_validate(item) for item in payload
        ]
        expected_ids = [question.question_id for question in batch.questions]
        actual_ids = [answer.question_id for answer in answers]
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("Batch response contains duplicate question IDs")
        if set(actual_ids) != set(expected_ids):
            raise ValueError("Batch response does not answer every question")
        if any(not answer.answer.strip() for answer in answers):
            raise ValueError("Batch response contains an empty answer")
        by_id = {answer.question_id: answer for answer in answers}
        return [by_id[question_id] for question_id in expected_ids]

    @staticmethod
    def __decode_json(response: str) -> object:
        """Decode JSON after removing an optional Markdown code fence."""
        content = response.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise ValueError("Incomplete JSON code fence")
            content = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Batch response is not valid JSON") from error
