from pathlib import Path
from pydantic import BaseModel
from typing import List
import json


def load_json_file(file: Path) -> dict:
    """ Open, read and return the content of a JSON file """
    with open(file, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content


def create_json_file(
        file_path: Path,
        content: List[BaseModel] | BaseModel) -> None:
    """
        Helper function to create a Json file from the given path and
            store the inputed content
    """
    data = [model.model_dump() for model in content]
    final: dict = {"search_results": data, "k": 10}
    with open(file_path, "w+") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    print(f"content save in {file_path}")


def create_answer_json_file(
        file_path: Path,
        content: List[BaseModel],
        k: int,
        ) -> None:
    """Store answer models using the answer-dataset JSON contract."""
    data = [model.model_dump(mode="json") for model in content]
    final: dict = {"answer_response": data, "k": k}
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_name(f".{file_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        print(f"storing answer in: '{file_path}'")
        json.dump(final, file, indent=4, ensure_ascii=False)
    temporary_path.replace(file_path)
