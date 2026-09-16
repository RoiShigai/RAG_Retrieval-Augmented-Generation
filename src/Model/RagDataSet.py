from pydantic import BaseModel
from typing import List
from .QuestionsModel import AnsweredQuestions, UnansweredQuestions


class RagDataSet(BaseModel):
    rag_questions: List[AnsweredQuestions | UnansweredQuestions]
