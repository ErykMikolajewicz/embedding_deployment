from collections.abc import Iterable
from typing import Protocol


class RestEmbeddingsAdapter(Protocol):
    def __init__(self, port: int, quantization: str): ...

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]: ...


class DirectEmbeddingsAdapter(Protocol):
    def __init__(self, quantization: str): ...

    def get_embeddings(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]: ...



class ContainerInstantiate(Protocol):
    def __init__(self, quantization: str):
       ...

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
