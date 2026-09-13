from ..Model import MinimalSource
from pathlib import Path


def retrieve_text_from_source(source: MinimalSource) -> str:
    """
        Helper Function to retrieve the text documentation from
            a MinimalSource Chunk data
    """
    with open(Path(source.file_path), "r") as f:
        f.seek(int(source.start), 1)
        content = f.read(int(source.end - source.start))
    return content
