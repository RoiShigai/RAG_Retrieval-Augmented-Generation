from Model import MinimalSource
from pathlib import Path


def retrieve_text_from_source(source: MinimalSource) -> str:
    """
        Helper Function to retrieve the text documentation from
            a MinimalSource Chunk data
    """
    with open(Path(source.file_path), "r", encoding="utf-8") as f:
        f.seek(int(source.first_character_index))
        content = f.read(
                int(source.last_character_index - source.first_character_index)
                )
    return content
