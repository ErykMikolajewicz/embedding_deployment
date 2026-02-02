from collections.abc import Iterable

import httpx


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
