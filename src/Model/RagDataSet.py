from pydantic import BaseModel


class RagDataSet(BaseModel):
    rag_questions: List[AnsweredQuestions | UnansweredQuestions]
