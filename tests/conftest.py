import pickle

import numpy as np
import pytest
from consts import QUANTIZATION_SIMILARITY_THRESHOLD


@pytest.fixture(scope="session")
def sentences() -> tuple[str, ...]:
    return (
        "Ala ma kota.",
        "Primus inter pares.",
        "vtefwTFGIFOUEPJIebyorwqpg9f3hurp[[qf3",
    )


def cosine_similarity(vector: list[float], reference_vector: list[float]):
    vector = np.array(vector)
    reference_vector = np.array(reference_vector)

    dot = np.dot(vector, reference_vector)
    norm_vec = np.linalg.norm(vector)
    norm_vec_ref = np.linalg.norm(reference_vector)
    result = dot / (norm_vec * norm_vec_ref)

    return result


@pytest.fixture(scope="session")
def measure_similarity(sentences):
    with open("./tests/data/reference_result.pickle", "rb") as f:
        reference_results = pickle.load(f)

    def _inner(results: list[list[float]]):
        for sentence, vector in zip(sentences, results, strict=True):
            reference_vector = reference_results[sentence]
            similarity = cosine_similarity(vector, reference_vector)
            assert similarity > QUANTIZATION_SIMILARITY_THRESHOLD, (
                f"Not enough similar to reference ({similarity}), for sentence:\n{sentence}"
            )

    return _inner
