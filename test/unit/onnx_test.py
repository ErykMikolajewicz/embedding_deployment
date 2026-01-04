import os

import pytest


@pytest.mark.parametrize("quantization", [None, 'fp16', 'int8', 'int4'])
def test_onnx_encoding(quantization, sentences, measure_similarity):
    os.environ['APP_DECODER_TYPE'] = 'ONNX'
    os.environ['APP_ENVIRONMENT'] = 'LOCAL_TEST'
    if quantization is not None:
        os.environ['QUANTIZATION'] = quantization
    else:
        try:
            del os.environ['QUANTIZATION']
        except KeyError:
            pass

    from src.infrastructure.adapters.onnx_encoding import encode

    vectors = encode(sentences)

    measure_similarity(vectors)