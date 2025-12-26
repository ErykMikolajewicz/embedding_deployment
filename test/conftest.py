import pytest
import numpy as np

@pytest.fixture(scope="session")
def sentences() -> tuple[str]:
    return ("Ala ma kota.", "Primus inter pares.", "vtefwTFGIFOUEPJIebyorwqpg9f3hurp[[qf3")


@pytest.fixture(scope="session")
def cosine_similarity():

    def _inner(a, b):
        results = []
        for v1, v2 in zip(a, b):
            v1 = np.array(v1)
            v2 = np.array(v2)

            dot = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            result = dot / (norm_v1 * norm_v2)
            results.append(result)

        return results
    return _inner