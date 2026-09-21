from typing import List, cast

import torch

from Helper import retrieve_text_from_source
from llm_sdk import Small_LLM_Model
from Model import MinimalSource


MAX_TOKEN: int = 128
STOP_MARKERS: tuple[str, ...] = (
    "<|im_end|>",
    "</s>",
    "<|endoftext|>",
)


class SLM:
    """Generate concise answers from retrieved context with a small model."""

    def __init__(
            self,
            model: Small_LLM_Model,
            max_token: int = MAX_TOKEN) -> None:
        """Initialize the SLM answer generator."""
        self.__model: Small_LLM_Model = model
        self.__max_token: int = max_token
        self.__stop_sequences: List[List[int]] = [
            cast(List[int], self.__model.encode(marker).tolist()[0])
            for marker in STOP_MARKERS
        ]
        self.__pre_prompt: str = (
                "<|im_start|>system\n"
                "You are a helpful and accurate AI assistant.\n"
                "Answer the user request using only the factual information"
                " in the context below.\n"
                "# RULES:\n"
                "1. Do not use outside knowledge or assumptions.\n"
                "2. If the answer is not in the context, say that the provided"
                " documents do not contain the answer.\n"
                "3. Cite the relevant source file when stating a fact.\n"
                "4. Be concise, clear, and professional.\n"
                "# RETRIEVED CONTEXT\n"
            )

    def generate_response(
            self,
            search_result: List[MinimalSource],
            query: str) -> str:
        """
        Generate an answer using the supplied retrieved context.

        Args:
            search_result: Sources containing the retrieved context.
            query: User request to answer.

        Returns:
            The generated answer text.
        """
        prompt: str = self.__build_prompt(query, search_result)
        return self.__generate(prompt)

    def __generate(self, prompt: str) -> str:
        """Generate tokens greedily until a stop marker or token limit."""
        input_ids: List[int] = cast(
            List[int], self.__model.encode(prompt).tolist()[0]
        )
        generated_ids: List[int] = []

        for _ in range(self.__max_token):
            logits = self.__model.get_logits_from_input_ids(input_ids)
            next_token: int = int(torch.argmax(torch.tensor(logits)).item())
            generated_ids.append(next_token)
            input_ids.append(next_token)
            stop_length = self.__stop_sequence_length(generated_ids)
            print(input_ids)
            if stop_length:
                generated_ids = generated_ids[:-stop_length]
                break

        return cast(str, self.__model.decode(generated_ids))

    def __stop_sequence_length(self, generated_ids: List[int]) -> int:
        """Return the length of a matching generated stop sequence."""
        for sequence in self.__stop_sequences:
            if sequence and generated_ids[-len(sequence):] == sequence:
                return len(sequence)
        return 0

    def __build_prompt(self, query: str, sources: List[MinimalSource]) -> str:
        """Build a direct-answer prompt from the retrieved sources."""
        context: str = ""

        for i, source in enumerate(sources):
            context += (
                f"SOURCE {i}:\n"
                f"File: {source.file_path}\n"
                f"{retrieve_text_from_source(source)}\n\n"
            )

        return (
            f"{self.__pre_prompt}{context}\n\n<|im_end|>\n"
            f"<|im_start|>user\n{query}\n/no_think<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
