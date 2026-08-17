from typing import List
from pydantic import BaseModel


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSources]
