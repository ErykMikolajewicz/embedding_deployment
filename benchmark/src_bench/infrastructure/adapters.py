from collections.abc import Iterable

import httpx
from src_bench.domain.enums import FrameworkType
from src_bench.domain.ports import EmbeddingsPort


def get_adapter_rest(adapter_type: FrameworkType) -> type[EmbeddingsPort]:
    match adapter_type:
        case FrameworkType.ONNX:
            return CustomRestAdapter
        case FrameworkType.OLLAMA:
            return OllamaAdapter
        case FrameworkType.SENTENCE_TRANSFORMERS:
            return CustomRestAdapter
        case _:
            raise Exception("Invalid adapter type!")


class CustomRestAdapter:
    def __init__(self, port: int, _: str):
        self.__port = port

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        url = f"http://localhost:{self.__port}/api/embed"

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

    def __init__(self, port: int, quantization: str):
        self.__port = port
        self.__model = OllamaAdapter.quantization_to_model[quantization]

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        url = f"localhost:{self.__port}/api/embed"

        payload = {
            "model": self.__model,
            "input": texts,
        }

        resp = httpx.post(url, json=payload, timeout=60)

        data = resp.json()

        embeddings = data["embeddings"]

        return embeddings


def get_direct_adapter(adapter_type: FrameworkType) -> EmbeddingsPort:
    match adapter_type:
        case FrameworkType.ONNX:
            return DirectOnnxAdapter
        case FrameworkType.OLLAMA:
            raise Exception("Ollama not supported")
        case FrameworkType.SENTENCE_TRANSFORMERS:
            return DirectSentenceTransformersAdapter
        case _:
            raise Exception("Invalid adapter type!")


class DirectOnnxAdapter:
    def __init__(self, quantization: str):
        from src.infrastructure.adapters.onnx_encoding import OnnxEncoder

        self.__encoder = OnnxEncoder(quantization)

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:

        embeddings = self.__encoder.encode(texts)

        return embeddings


class DirectSentenceTransformersAdapter:
    def __init__(self, quantization: str):
        from src.infrastructure.adapters.sentence_transformers_encoding import SentenceTransformersEncoder

        self.__encoder = SentenceTransformersEncoder(quantization)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.__encoder.encode(texts)

        return embeddings
