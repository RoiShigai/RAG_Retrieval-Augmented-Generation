from pathlib import Path
from pydantic import BaseModel
from typing import List
import json
import os


def load_json_file(file: Path) -> dict:
    """ Open, read and return the content of a JSON file """
    with open(file, "r", encoding="utf-8") as f:
        content = json.load(f)
    print(content)
    return content


def create_json_file(
        file_path: Path,
        content: List[BaseModel] | BaseModel) -> None:
    """
        Helper function to create a Json file from the given path and
            store the inputed content
    """
    data = {model.question_id: model.model_dump() for model in content}
    with open(file_path, "w+") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
