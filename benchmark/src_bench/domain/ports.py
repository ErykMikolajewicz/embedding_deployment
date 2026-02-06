from collections.abc import Callable, Sequence
from typing import Protocol

from src_bench.domain.enums import Quantization
from src_bench.domain.types import Embeddings

MeasureFunction = Callable[[Sequence[str]], Embeddings]


class DirectAdapter(Protocol):
    def __init__(self, quantization: Quantization): ...

    def get_embeddings(self, texts: Sequence[str]) -> Embeddings: ...


class RestAdapter(Protocol):
    def __init__(self, port: int, quantization: Quantization): ...

    def get_embeddings(self, texts: Sequence[str]) -> Embeddings: ...


class EnvironmentPreparator(Protocol):
    def __init__(self, port: int, quantization: Quantization): ...

    def __enter__(self): ...

    def __exit__(self, exc_type, exc_val, exc_tb): ...
