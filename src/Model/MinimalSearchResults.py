from typing import List
from .MinimalSource import MinimalSource
from pydantic import BaseModel


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str
