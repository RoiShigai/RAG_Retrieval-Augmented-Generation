from typing import List, cast

from Helper import retrieve_text_from_source
from llm_sdk import Small_LLM_Model
from Model import MinimalSource
from .AnswerBatch import (
    AnswerBatch,
    BatchQuestion,
    ContextBatchBuilder,
)
from Model import MinimalAnswer


MAX_TOKEN: int = 64
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
        self.__question: int = 0

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

    def build_batch_builder(
            self,
            max_questions: int = 10,
            max_char_length: int = 8192,
            ) -> ContextBatchBuilder:
        """Return a context builder using rendered character limits."""
        return ContextBatchBuilder(
            max_questions=max_questions,
            max_char_length=max_char_length,
        )

    def generate_batch(self, batch: AnswerBatch) -> list[MinimalAnswer]:
        """Generate validated answers for one shared-context batch."""
        context_prompt = self.__build_batch_context(batch)
        question_prompts = {
            question.question_id: self.__build_question_prompt(question)
            for question in batch.questions
        }
        generated = self.__model.generate_batch(
            [context_prompt], question_prompts, self.__max_token
        )
        if set(generated) != set(question_prompts):
            raise ValueError("Batch generation did not answer every question")
        sources = {chunk.source_id: chunk.source for chunk in batch.chunks}
        questions = {
            question.question_id: question for question in batch.questions
        }
        return [
            MinimalAnswer.model_construct(
                question_id=question_id,
                question=questions[question_id].question,
                retrieved_sources=[
                    sources[source_id]
                    for source_id in questions[question_id].source_ids
                ],
                answer=generated[question_id],
            )
            for question_id in questions
        ]

    def __generate(self, prompt: str) -> str:
        """Generate and decode an answer using cached model inference."""
        print(f"question {self.__question}")
        prompt_ids = self.__model.encode(prompt)
        generated_ids = self.__model.generate(
            prompt_ids,
            self.__max_token,
            None,
            self.__stop_sequences,
        )
        return self.__model.decode(generated_ids)

    def __generate_individually(
            self,
            batch: AnswerBatch,
            ) -> list[MinimalAnswer]:
        """Fallback to isolated generation when batch output is invalid."""
        sources = {chunk.source_id: chunk.source for chunk in batch.chunks}
        answers: list[MinimalAnswer] = []
        for question in batch.questions:
            answer = self.generate_response(
                [sources[source_id] for source_id in question.source_ids],
                question.question,
            )
            answers.append(MinimalAnswer.model_construct(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    sources[source_id] for source_id in question.source_ids
                ],
                answer=answer,
            ))
        return answers

    def __build_batch_context(self, batch: AnswerBatch) -> str:
        """Build the context prompt that is cached once per batch."""
        context = "\n\n".join(
            f"[{chunk.source_id}]\nFile: {chunk.source.file_path}\n"
            f"{chunk.text}"
            for chunk in batch.chunks
        )
        return (
            f"{self.__pre_prompt}{context}\n\n<|im_end|>\n"
        )

    def __build_question_prompt(self, question: BatchQuestion) -> str:
        """Build one uncoupled question prompt for cached context."""
        return (
            "<|im_start|>user\n"
            f"Allowed sources: {', '.join(question.source_ids)}\n"
            f"Question: {question.question}\n"
            "/no_think<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

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
