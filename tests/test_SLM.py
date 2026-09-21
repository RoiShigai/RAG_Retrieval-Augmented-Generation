from typing import List

import torch

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
            stop_sequences: List[List[int]]) -> List[int]:
        generated: List[int] = []
        for token in self.next_tokens[:max_new_tokens]:
            self.calls += 1
            generated.append(token)
            for sequence in stop_sequences:
                if sequence and generated[-len(sequence):] == sequence:
                    return generated[:-len(sequence)]
        return generated

    def decode(self, ids: List[int]) -> str:
        return "answer: " + ",".join(str(token) for token in ids)


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
