from Model import MinimalSource
from pathlib import Path


def retrieve_text_from_source(source: MinimalSource) -> str:
    """
        Helper Function to retrieve the text documentation from
            a MinimalSource Chunk data
    """
    print(source)
    with open(Path(source.file_path), "rb") as f:
        f.seek(int(source.first_character_index), 1)
        content = f.read(
                int(source.last_character_index - source.first_character_index)
                )
    return content
