import pytest

from src.infrastructure.adapters.onnx_encoding import encode


@pytest.mark.parametrize("quantization", ['', 'fp16', 'int8', 'int4'])
def test_onnx_encoding(quantization, sentences, measure_similarity):
    vectors = encode(sentences)

    measure_similarity(vectors)