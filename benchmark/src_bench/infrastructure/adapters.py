from collections.abc import Iterable

import httpx

from src.infrastructure.exceptions import AdapterNotSupported, InvalidConfigValue
from src_bench.domain.enums import AdapterType, FrameworkType
from src_bench.domain.ports import EmbeddingsAdapter


def get_adapter(framework_type: FrameworkType, adapter_type: AdapterType) -> type[EmbeddingsAdapter]:
    match framework_type, adapter_type:
        case FrameworkType.ONNX, AdapterType.REST:
            return CustomRestAdapter
        case FrameworkType.OLLAMA, AdapterType.REST:
            return OllamaAdapter
        case FrameworkType.SENTENCE_TRANSFORMERS, AdapterType.REST:
            return CustomRestAdapter
        case FrameworkType.ONNX, AdapterType.DIRECT:
            return DirectOnnxAdapter
        case FrameworkType.OLLAMA, AdapterType.DIRECT:
            raise AdapterNotSupported("direct adapter", FrameworkType.OLLAMA)
        case FrameworkType.SENTENCE_TRANSFORMERS, AdapterType.DIRECT:
            return DirectSentenceTransformersAdapter
        case _:
            raise InvalidConfigValue("framework type, adapter type", f"{framework_type}, {adapter_type}")


class CustomRestAdapter:
    def __init__(self, port: int, _: str):
        self.__port = port

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        url = f"http://localhost:{self.__port}/api/embed"

        payload = texts

        resp = httpx.post(url, json=payload, timeout=500)

        embeddings = resp.json()

        return embeddings


class OllamaAdapter:
    quantization_to_model = {
        None: "embeddinggemma:300m",
        "bf16": "embeddinggemma:300m-bf16",
        "int8": "embeddinggemma:300m-qat-q8_0",
        "int4": "embeddinggemma:300m-qat-q4_0",
    }

    def __init__(self, port: int, quantization: str):
        self.__port = port
        self.__model = OllamaAdapter.quantization_to_model[quantization]

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        url = f"http://localhost:{self.__port}/api/embed"

        payload = {
            "model": self.__model,
            "input": texts,
        }

        resp = httpx.post(url, json=payload, timeout=60)

        data = resp.json()

        embeddings = data["embeddings"]

        return embeddings


class DirectOnnxAdapter:
    def __init__(self, _: int, quantization: str):
        from src.infrastructure.adapters.onnx_encoding import OnnxEncoder

        self.__encoder = OnnxEncoder(quantization)

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        embeddings = self.__encoder.encode(texts)

        return embeddings


class DirectSentenceTransformersAdapter:
    def __init__(self, _: int, quantization: str):
        from src.infrastructure.adapters.sentence_transformers_encoding import SentenceTransformersEncoder

        self.__encoder = SentenceTransformersEncoder(quantization)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.__encoder.encode(texts)

        return embeddings
