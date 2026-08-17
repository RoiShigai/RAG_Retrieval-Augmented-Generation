from MinimalSource import MinimalSource
from pydantic import BaseModel, Field
from typing import List
import uuid


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid64()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
