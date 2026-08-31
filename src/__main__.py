from .Indexor import Indexor
from pathlib import Path

indexor = Indexor.Indexor()
indexor.index(Path("vllm-0.10.1"))
