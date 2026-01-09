from unittest.mock import patch

import pytest

from src.infrastructure.adapters.onnx_encoding import OnnxEncoder


@pytest.mark.parametrize("quantization", [None, "fp16", "int8", "int4"])
def test_onnx_encoding(quantization, sentences, measure_similarity):

    with patch("src.infrastructure.adapters.onnx_encoding.MODEL_ROOT", "./models"):
        onnx_encoder = OnnxEncoder(quantization)

    vectors = onnx_encoder.encode(sentences)

    measure_similarity(vectors)
