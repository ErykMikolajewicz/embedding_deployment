from unittest.mock import patch

import pytest

from src.infrastructure.adapters.sentence_transformers_encoding import SentenceTransformersEncoder


@pytest.mark.parametrize("quantization", [None, "bf16"])
def test_sentence_transformers_encoding(quantization, sentences, measure_similarity):
    sentences = list(sentences)  # to satisfy type hints

    with patch(
        "src.infrastructure.adapters.sentence_transformers_encoding.get_model_root_path", return_value="./models"
    ):
        encoder = SentenceTransformersEncoder(quantization)
    vectors = encoder.encode(sentences)

    measure_similarity(vectors)
