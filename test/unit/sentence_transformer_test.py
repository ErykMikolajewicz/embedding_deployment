import os

import pytest


@pytest.mark.parametrize("quantization", [None, 'int8', 'int4'])
def test_sentence_transformers_encoding(quantization, sentences, measure_similarity):
    os.environ['APP_DECODER_TYPE'] = 'SENTENCE_TRANSFORMERS'
    os.environ['APP_ENVIRONMENT'] = 'LOCAL_TEST'
    if quantization is not None:
        os.environ['QUANTIZATION'] = quantization
    else:
        try:
            del os.environ['QUANTIZATION']
        except KeyError:
            pass

    from src.infrastructure.adapters.sentence_transformers_encoding import encode

    sentences = list(sentences) # to satisfy type hints
    vectors = encode(sentences)

    measure_similarity(vectors)