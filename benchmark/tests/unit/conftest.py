import pytest


@pytest.fixture(scope="session")
def sentences() -> tuple[str, ...]:
    return (
        "Ala ma kota.",
        "Primus inter pares.",
        "vtefwTFGIFOUEPJIebyorwqpg9f3hurp[[qf3",
    )


@pytest.fixture
def fake_embeddings() -> list[list[float]]:
    return [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
