from collections.abc import Iterable
from typing import Any

import httpx
from src_bench.domain.enums import FrameworkType


def get_adapter(adapter_type: FrameworkType) -> Any:
    match adapter_type:
        case FrameworkType.ONNX:
            return CustomFormatAdapter
        case FrameworkType.OLLAMA:
            return OllamaAdapter
        case FrameworkType.SENTENCE_TRANSFORMERS:
            return CustomFormatAdapter
        case _:
            raise Exception("Invalid adapter type!")


class CustomFormatAdapter:
    def __init__(self, port: int):
        self.__port = port

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        url = f"localhost:{self.__port}/api/embed"

        payload = texts

        resp = httpx.post(url, json=payload, timeout=60)

        embeddings = resp.json()

        return embeddings


class OllamaAdapter:
    quantization_to_model = {
        None: "embeddinggemma:300m",
        "bf16": "embeddinggemma:300m-bf16",
        "int8": "embeddinggemma:300m-qat-q8_0",
        "int4": "embeddinggemma:300m-qat-q4_0",
    }

    def __init__(self, port: int):
        self.__port = port

    def get_embeddings(self, texts: Iterable[str], quantization: str) -> list[list[float]]:
        url = f"localhost:{self.__port}/api/embed"

        model = OllamaAdapter.quantization_to_model[quantization]

        payload = {
            "model": model,
            "input": texts,
        }

        resp = httpx.post(url, json=payload, timeout=60)

        data = resp.json()

        embeddings = data["embeddings"]

        return embeddings
