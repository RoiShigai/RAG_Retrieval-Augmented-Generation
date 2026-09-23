from typing import List

import torch

from src.SLM.AnswerBatch import AnswerBatch, BatchQuestion
from src.SLM.SLM import SLM


class FakeModel:
    """Small fake model for testing the SLM generation loop."""

    def __init__(self, next_tokens: List[int]) -> None:
        self.next_tokens = next_tokens
        self.calls = 0
        self.encoded_text: List[str] = []

    def encode(self, text: str) -> torch.Tensor:
        self.encoded_text.append(text)
        marker_ids = {
            "<|im_end|>": [90],
            "</s>": [91],
            "<|endoftext|>": [92],
        }
        return torch.tensor([marker_ids.get(text, [1])])

    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            past_key_values: object = None,
            stop_sequences: List[List[int]] | None = None) -> List[int]:
        generated: List[int] = []
        if stop_sequences is None:
            stop_sequences = []
        for token in self.next_tokens[:max_new_tokens]:
            self.calls += 1
            generated.append(token)
            for sequence in stop_sequences:
                if sequence and generated[-len(sequence):] == sequence:
                    return generated[:-len(sequence)]
        return generated

    def decode(self, ids: List[int]) -> str:
        return "answer: " + ",".join(str(token) for token in ids)


class FakeBatchModel(FakeModel):
    """Record cached-context batch calls and return raw answers."""

    def __init__(self) -> None:
        super().__init__([])
        self.batch_calls: List[tuple[List[str], dict[str, str], int]] = []

    def generate_batch(
            self,
            prompts: List[str],
            questions: dict[str, str],
            max_new_tokens: int,
            ) -> dict[str, str]:
        self.batch_calls.append((prompts, questions, max_new_tokens))
        return {
            question_id: f"answer for {question_id}"
            for question_id in questions
        }


def test_slm_uses_direct_answer_prompt_and_stop_marker() -> None:
    model = FakeModel([10, 11, 90, 99])
    slm = SLM(model, max_token=128)  # type: ignore[arg-type]

    answer = slm.generate_response([], "What is the answer?")

    prompt = model.encoded_text[3]
    assert "<|think|>" not in prompt
    assert "/no_think" in prompt
    assert answer == "answer: 10,11"
    assert model.calls == 3


def test_slm_respects_configured_token_limit() -> None:
    model = FakeModel([10, 11, 12, 13])
    slm = SLM(model, max_token=3)  # type: ignore[arg-type]

    answer = slm.generate_response([], "What is the answer?")

    assert answer == "answer: 10,11,12"
    assert model.calls == 3


def test_slm_batches_questions_over_one_cached_context() -> None:
    model = FakeBatchModel()
    slm = SLM(model, max_token=7)  # type: ignore[arg-type]
    batch = AnswerBatch(
        chunks=[],
        questions=[
            BatchQuestion(
                question_id="q1",
                question="What is one?",
                source_ids=[],
            ),
            BatchQuestion(
                question_id="q2",
                question="What is two?",
                source_ids=[],
            ),
        ],
    )

    answers = slm.generate_batch(batch)

    assert len(model.batch_calls) == 1
    context_prompts, question_prompts, max_tokens = model.batch_calls[0]
    assert len(context_prompts) == 1
    assert "What is one?" not in context_prompts[0]
    assert "What is two?" not in context_prompts[0]
    assert "JSON" not in context_prompts[0]
    assert set(question_prompts) == {"q1", "q2"}
    assert "What is one?" in question_prompts["q1"]
    assert "What is two?" in question_prompts["q2"]
    assert max_tokens == 7
    assert [(answer.question_id, answer.answer) for answer in answers] == [
        ("q1", "answer for q1"),
        ("q2", "answer for q2"),
    ]
