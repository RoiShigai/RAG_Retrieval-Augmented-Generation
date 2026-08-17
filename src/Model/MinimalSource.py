from pydantic import BaseModel
from locale import str


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
