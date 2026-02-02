from src_bench.domain.services.data import get_benchmark_data


def test_get_benchmark_data():
    benchmark_data = get_benchmark_data()

    assert isinstance(benchmark_data, tuple)
    assert all(isinstance(text, str) for text in benchmark_data)
