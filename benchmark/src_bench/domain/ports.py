from typing import Protocol, Any
from collections.abc import Callable


class EmbeddingsPort(Protocol):
    __init__: Callable[..., Any]

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        ...