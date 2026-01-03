import os

import pytest

from src.infrastructure.adapters.sentence_transformers_encoding import encode


@pytest.mark.parametrize("quantization", [''])
def test_sentence_transformers_encoding(quantization, sentences, measure_similarity):
    os.environ['APP_DECODER_TYPE'] = 'SENTENCE_TRANSFORMERS'
    os.environ['APP_ENVIRONMENT'] = 'LOCAL_TEST'
    os.environ['QUANTIZATION'] = quantization

    sentences = list(sentences) # to satisfy type hints
    vectors = encode(sentences)

    measure_similarity(vectors)