# from unittest.mock import patch
#
# import httpx
# import pytest
#
# import src_bench.infrastructure.adapters as adapters
# from src.infrastructure.exceptions import AdapterNotSupported
# from src_bench.consts import DEFAULT_HTTP_PORT
# from src_bench.domain.enums import AdapterType, FrameworkType
#
#
# @pytest.mark.parametrize("framework_type", FrameworkType)
# def test_get_rest_adapter(framework_type: FrameworkType):
#     adapter_class = adapters.get_adapter(framework_type, AdapterType.REST)
#
#     assert callable(getattr(adapter_class, "get_embeddings", None))
#
#
# def test_custom_rest_adapter(sentences, fake_embeddings):
#     custom_format_adapter = adapters.CustomRestAdapter(DEFAULT_HTTP_PORT, "")
#
#     response = httpx.Response(
#         status_code=200,
#         json=fake_embeddings,
#         request=httpx.Request("POST", f"http://localhost:{DEFAULT_HTTP_PORT}/api/embed"),
#     )
#
#     with patch("httpx.post", return_value=response):
#         embeddings = custom_format_adapter.get_embeddings(sentences)
#
#     assert len(embeddings) == len(sentences)
#
#     for embedding in embeddings:
#         assert isinstance(embedding, list)
#         assert all(isinstance(number, float) for number in embedding)
#
#
# @pytest.mark.parametrize("quantization", [None, "bf16", "int8", "int4"])
# def test_ollama_adapter(sentences, fake_embeddings, quantization):
#     ollama_adapter = adapters.OllamaAdapter(DEFAULT_HTTP_PORT, quantization)
#
#     ollama_json = {"embeddings": fake_embeddings}
#
#     response = httpx.Response(
#         status_code=200,
#         json=ollama_json,
#         request=httpx.Request("POST", f"http://localhost:{DEFAULT_HTTP_PORT}/api/embed"),
#     )
#
#     with patch("httpx.post", return_value=response):
#         embeddings = ollama_adapter.get_embeddings(sentences)
#
#     assert len(embeddings) == len(sentences)
#
#     for embedding in embeddings:
#         assert isinstance(embedding, list)
#         assert all(isinstance(number, float) for number in embedding)
#
#
# @pytest.mark.parametrize("framework_type", FrameworkType)
# def test_get_direct_adapter(framework_type: FrameworkType):
#     if framework_type == FrameworkType.OLLAMA:
#         with pytest.raises(AdapterNotSupported):
#             adapters.get_adapter(framework_type, AdapterType.DIRECT)
#     else:
#         adapter_class = adapters.get_adapter(framework_type, AdapterType.DIRECT)
#         assert callable(getattr(adapter_class, "get_embeddings", None))
#
#
# @pytest.mark.parametrize("quantization", [None, "fp16", "int8", "int4"])
# def test_direct_onnx_adapter(sentences, quantization):
#     with patch("src.infrastructure.adapters.onnx_encoding.get_model_root_path", return_value="./models"):
#         direct_onnx_adapter = adapters.DirectOnnxAdapter(DEFAULT_HTTP_PORT, quantization)
#
#     embeddings = direct_onnx_adapter.get_embeddings(sentences)
#
#     assert len(embeddings) == len(sentences)
#
#     for embedding in embeddings:
#         assert isinstance(embedding, list)
#         assert all(isinstance(number, float) for number in embedding)
#
#
# @pytest.mark.parametrize("quantization", [None, "bf16"])
# def test_direct_sentence_transformers_adapter(sentences, quantization):
#     with patch(
#         "src.infrastructure.adapters.sentence_transformers_encoding.get_model_root_path", return_value="./models"
#     ):
#         direct_st_adapter = adapters.DirectSentenceTransformersAdapter(DEFAULT_HTTP_PORT, quantization)
#
#     sentences = list(sentences)  # to satisfy type hints
#     embeddings = direct_st_adapter.get_embeddings(sentences)
#
#     assert len(embeddings) == len(sentences)
#
#     for embedding in embeddings:
#         assert isinstance(embedding, list)
#         assert all(isinstance(number, float) for number in embedding)
