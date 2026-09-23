# ABOUTME: LLM SDK for local Hugging Face model inference.
# ABOUTME: Provides Small_LLM_Model for causal language model inference.

from typing import Any, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
)
from huggingface_hub import hf_hub_download


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Wrap a lightweight Hugging Face causal-LM for local inference.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None*, select MPS, CUDA, or CPU automatically.
    dtype: torch.dtype | None, default=None
        Numerical precision. GPU and MPS default to ``float16``; CPU uses
        ``float32``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = (
                torch.float16
                if self._device in ["cuda", "mps"]
                else torch.float32
            )
        self._dtype = dtype

        # Load tokenizer and model.
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        if self._device != "cuda":
            self._model.to(self._device)
        self._model.eval()
        self._model.config.use_cache = True

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise text into a 2-D input ID tensor on the target device."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return cast(str, self._tokenizer.decode(ids, skip_special_tokens=True))

    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            past_key_values: Any = None,
            stop_sequences: list[list[int]] | None = None) -> list[int]:
        """Generate tokens with a cached key-value state."""
        if max_new_tokens <= 0:
            return []
        if stop_sequences is None:
            stop_sequences = []

        with torch.inference_mode():
            outputs = self._model(
                input_ids=prompt_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            cached_past_key_values: Any = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_ids: list[int] = []

            for _ in range(max_new_tokens):
                token_id = int(next_token.item())
                generated_ids.append(token_id)
                stop_length = self._stop_sequence_length(
                    generated_ids, stop_sequences
                )
                if stop_length:
                    generated_ids = generated_ids[:-stop_length]
                    break

                outputs = self._model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=cached_past_key_values,
                    use_cache=True,
                )
                cached_past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        return generated_ids

    def generate_batch(
            self,
            prompts: list[str],
            questions: dict[str, str],
            max_new_tokens: int,
            ) -> dict[str, str]:
        """Generate raw answers using one cached context prompt."""
        answers: dict[str, str] = {}
        if not prompts or not questions or max_new_tokens <= 0:
            return answers
        if len(prompts) != 1:
            raise ValueError("generate_batch expects one shared context")
        past_key_values = self.__cache_context(prompts)
        stop_sequences = [
                self._tokenizer.encode(seq, add_special_tokens=False)
                for seq in ["<|im_end|>", "</s>", "<|endoftext|>"]
            ]

        for question_id, question in questions.items():
            print(question)
            question_ids = self._tokenizer(
                    question,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(self._device)
            answer_ids = self.generate(
                    question_ids,
                    max_new_tokens,
                    past_key_values,
                    stop_sequences
                )
            answers[question_id] = cast(
                str,
                self._tokenizer.decode(answer_ids, skip_special_tokens=True),
            )

        return answers

    def __cache_context(self, context: list[str]):
        """ Encode and Cache the context into the Model KV Cache """
        context_id = self._tokenizer(
            context,
            return_tensors="pt",
        )["input_ids"].to(self._device)

        with torch.inference_mode():
            outputs = self._model(
                    input_ids=context_id,
                    use_cache=True
                    )
        return outputs.past_key_values

    @staticmethod
    def _stop_sequence_length(
            generated_ids: list[int],
            stop_sequences: list[list[int]]) -> int:
        """Return the length of a matching stop sequence."""
        for sequence in stop_sequences:
            if sequence and generated_ids[-len(sequence):] == sequence:
                return len(sequence)
        return 0

    def get_path_to_vocab_file(self) -> str:
        """Return the downloaded tokenizer vocabulary path."""
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            'vocab_file', "vocab.json"
        )
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return cast(str, vocab_path)

    def get_path_to_merges_file(self) -> str:
        """Return the downloaded tokenizer merges path."""
        merges_file_name = self._tokenizer.vocab_files_names.get(
            'merges_file', "merges.txt"
        )
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return cast(str, merges_path)

    def get_path_to_tokenizer_file(self) -> str:
        """Return the downloaded tokenizer JSON path."""
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            'tokenizer_file', "tokenizer.json"
        )
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return cast(str, tokenizer_path)
