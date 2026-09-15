from llm_sdk import Small_LLM_Model
from Model import MinimalSource
from Helper import retrieve_text_from_source
from typing import List, cast
import torch


STOP_WORD: str = "\0"
MAX_TOKEN: int = 60


class SLM:
    """
        SLM class definition.

        The SLM will be the small language will generate a
            formatted response to a user query by ingesting the
            the retrieved context from the RAG retriever.
    """

    def __init__(
            self,
            model: Small_LLM_Model,
            max_token: int = MAX_TOKEN) -> None:
        """ Init method for the SLM """
        self.__model: Small_LLM_Model = model
        self.__eos_token_id: int = self.__model.encode("\0").tolist()[0][0]
        self.__max_token: int = max_token
        self.__pre_prompt: str = (
                "<|im_start|>system\n"
                "You are an helpfull and accurate AI assistant."
                "Your task is to answer the UserRequest using only the"
                " factual informations provided in the context below.\n"
                "# RULES:\n"
                "1. Grounding: Do not use any outside knowledge or assumption."
                " If the answer cannot be found in the context, state: '"
                "I cannot respond with the provided documents'.\n"
                "2. Citation: Cite the specific source or documentation"
                "title when stating a fact.\n"
                "3. Tone: Be concise, clear and professional."
                " Do not speculate or exptrapolate beyond the text."
                f"4. Stop word: use this stop word '{STOP_WORD}'"
                "when you finish your answer."
                "# RETRIVED CONTEXT\n"
            )

    def generate_response(
            self,
            search_result: List[MinimalSource],
            query: str) -> str:
        """
            Generate a formatted response using the retrieved context
                given by the search result for the given query.

            Parameters:
                search_result: List[MinimalSource] | the context retrieved
                query: str | the user query to answer

            Return:
                str
        """
        prompt: str = self.__build_prompt(
                query,
                search_result
            )
        answer: str = self.__generate(prompt)
        return answer

    def __generate(self, prompt: str) -> str:
        """
            Generation loop of the SLM.

            Will produce an answer string of MAX_TOKEN tokens.
        """

        input_ids: List[int] = self.__model.encode(prompt).tolist()[0]
        generated_ids: List[int] = []
        print(prompt)

        for _ in range(self.__max_token):
            logits = self.__model.get_logits_from_input_ids(
                input_ids
            )
            next_token: int = int(torch.argmax(
                torch.tensor(logits)
            ).item())
            if next_token == self.__eos_token_id:
                break
            generated_ids.append(next_token)
            input_ids.append(next_token)
            print(generated_ids)

        return cast(str, self.__model.decode(generated_ids))

    def __build_prompt(self, query: str, sources: List[MinimalSource]) -> str:
        """ Generate the correct prompt for the SLM Answer generation """
        context: str = ""

        for i, source in enumerate(sources):
            context += (
                        f"SOURCE {i}:\n"
                        f"File: {source.file_path}\n"
                        f"{retrieve_text_from_source(source)}\n\n"
                        )

        return (f"{self.__pre_prompt}{context}\n\n<|im_end|>\n"
                f"<|im_start|>user\n {query}\n<|im_end|>\n"
                "<|im_start|>assistant\n<|think|>\n"
                "# ANSWER: "
                )
