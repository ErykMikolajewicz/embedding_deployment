"""That etest was written before was observed, that llama-cpp-python not work with gemma300m
It's actually functional, and waiting for time, when support for gemma will be added.
Issue of that problem on GitHub: https://github.com/abetlen/llama-cpp-python/issues/2065"""

# from unittest.mock import patch
#
# import pytest
#
# from src.infrastructure.adapters.llama_cpp_encoding import LlamaCppEncoder
#
#
# @pytest.mark.parametrize("quantization", [None, "bf16", "int8", "int4"])
# def test_llama_cpp_encoding(quantization, sentences, measure_similarity):
#
#     with patch("src.infrastructure.adapters.llama_cpp_encoding.get_model_root_path", return_value="./models"):
#         onnx_encoder = LlamaCppEncoder(quantization)
#
#     sentences = list(sentences) # to satisfy type hints
#     vectors = onnx_encoder.encode(sentences)
#
#     measure_similarity(vectors)
