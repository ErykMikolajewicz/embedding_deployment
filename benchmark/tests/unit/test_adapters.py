from unittest.mock import patch

import httpx
import pytest

import src_bench.infrastructure.adapters as adapters
from benchmark.tests.consts import HTTP_PORT
from src.domain.enums import Quantization


def test_custom_rest_adapter(sentences, fake_embeddings):
    custom_format_adapter = adapters.CustomRestAdapter(HTTP_PORT, None)

    response = httpx.Response(
        status_code=200,
        json=fake_embeddings,
        request=httpx.Request("POST", f"http://localhost:{HTTP_PORT}/api/embed"),
    )

    with patch("httpx.post", return_value=response):
        embeddings = custom_format_adapter.get_embeddings(sentences)

    assert len(embeddings) == len(sentences)

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert all(isinstance(number, float) for number in embedding)


def test_ollama_adapter(sentences, fake_embeddings):
    ollama_adapter = adapters.OllamaAdapter(HTTP_PORT, None)

    ollama_json = {"embeddings": fake_embeddings}

    response = httpx.Response(
        status_code=200,
        json=ollama_json,
        request=httpx.Request("POST", f"http://localhost:{HTTP_PORT}/api/embed"),
    )

    with patch("httpx.post", return_value=response):
        embeddings = ollama_adapter.get_embeddings(sentences)

    assert len(embeddings) == len(sentences)

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert all(isinstance(number, float) for number in embedding)


onnx_quantization = (None, Quantization.FP16, Quantization.INT8, Quantization.INT4)


@pytest.mark.parametrize("quantization", onnx_quantization)
def test_direct_onnx_adapter(sentences, quantization):
    with patch("src.infrastructure.adapters.onnx_encoding.get_model_root_path", return_value="./models"):
        direct_onnx_adapter = adapters.DirectOnnxAdapter(quantization)

    embeddings = direct_onnx_adapter.get_embeddings(sentences)

    assert len(embeddings) == len(sentences)

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert all(isinstance(number, float) for number in embedding)


sentence_transformers_quantization = (None, Quantization.BF16)


@pytest.mark.parametrize("quantization", sentence_transformers_quantization)
def test_direct_sentence_transformers_adapter(sentences, quantization):
    with patch(
        "src.infrastructure.adapters.sentence_transformers_encoding.get_model_root_path", return_value="./models"
    ):
        direct_st_adapter = adapters.DirectSentenceTransformersAdapter(quantization)

    embeddings = direct_st_adapter.get_embeddings(sentences)

    assert len(embeddings) == len(sentences)

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert all(isinstance(number, float) for number in embedding)
