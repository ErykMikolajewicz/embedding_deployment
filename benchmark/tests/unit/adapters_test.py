from unittest.mock import patch

import httpx
import pytest
from src_bench.domain.enums import FrameworkType
from src_bench.infrastructure.adapters import CustomRestAdapter, OllamaAdapter, get_adapter_rest


@pytest.mark.parametrize("framework_type", FrameworkType)
def test_get_adapter(framework_type: FrameworkType):
    adapter_class = get_adapter_rest(framework_type)

    assert callable(getattr(adapter_class, "get_embeddings"))


def test_custom_rest_adapter(sentences, fake_embeddings):
    port = 8000
    custom_format_adapter = CustomRestAdapter(port, "")

    response = httpx.Response(
        status_code=200,
        json=fake_embeddings,
        request=httpx.Request("POST", f"http://localhost:{port}/api/embed"),
    )

    with patch("httpx.post", return_value=response):
        embeddings = custom_format_adapter.get_embeddings(sentences)

    assert len(embeddings) == len(sentences)

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert all(isinstance(number, float) for number in embedding)


@pytest.mark.parametrize("quantization", [None, "bf16", "int8", "int4"])
def test_ollama_adapter(sentences, fake_embeddings, quantization):
    port = 8000
    ollama_adapter = OllamaAdapter(port, quantization)

    ollama_json = {"embeddings": fake_embeddings}

    response = httpx.Response(
        status_code=200,
        json=ollama_json,
        request=httpx.Request("POST", f"http://localhost:{port}/api/embed"),
    )

    with patch("httpx.post", return_value=response):
        embeddings = ollama_adapter.get_embeddings(sentences)

    assert len(embeddings) == len(sentences)

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert all(isinstance(number, float) for number in embedding)
