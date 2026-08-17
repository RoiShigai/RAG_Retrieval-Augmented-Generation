from pydantic import BaseModel


class ChunkModel(BaseModel):
    path_file: str
    chunk_id: int
    start_char: int
    end_char: int
