from .MinimalSearchResults import MinimalSearchResults, MinimalAnswer
from pydantic import BaseModel
from typing import List


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
