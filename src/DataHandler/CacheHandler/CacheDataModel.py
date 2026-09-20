from pydantic import BaseModel
from Model import MinimalAnswer, MinimalSource


class CacheDataModel(BaseModel):
    """Represent one persisted search or answer cache entry."""

    query: str
    k: int
    database_version: str
    search_result: list[MinimalSource] | None = None
    answer: MinimalAnswer | None = None
