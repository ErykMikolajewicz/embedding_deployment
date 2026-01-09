from collections.abc import Iterable
from typing import Protocol


class EmbeddingsPort(Protocol):
    def __init__(self, port: int, quantization: str): ...

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]: ...
