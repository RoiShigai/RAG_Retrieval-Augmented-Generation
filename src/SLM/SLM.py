from llm_sdk import Small_LLM_Model
from ..Model import MinimalAnswer, MinimalSource
from ..Helper import retrieve_text_from_source
from typing import List


MAX_TOKEN: int = 500


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
                "# RETRIVED CONTEXT\n"
            )

    def generate_response(
            search_result: List[MinimalSource],
            query: str) -> MinimalAnswer:
        """
            Generate a formatted response using the retrieved context
                given by the search result for the given query.

            Parameters:
                search_result: List[MinimalSource] | the context retrieved
                query: str | the user query to answer

            Return:
                MinimalAnswer
        """

    def __build_prompt(self, query: str, context: List[MinimalSource]) -> str:
        """ Generate the correct prompt for the SLM Answer generation """
        context: str = ""

        for i, source in enumerate(context):
            context += (
                    f"SOURCE {i}:\n"
                    f"File: {source.file_path}"
                    f"{retrieve_text_from_source(source)}\n\n"
                )
        return f"{self.__pre_prompt}{context}\n\n#UserQuery: {query}\n# ANSWER: "
