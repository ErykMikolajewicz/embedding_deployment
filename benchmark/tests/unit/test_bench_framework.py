from unittest.mock import Mock

from src_bench.domain.services.benchmarking import BenchmarkRunner


def test_benchmark_framework(sentences):
    measure_function = Mock(return_value=None)

    benchmark_runner = BenchmarkRunner(measure_number=5, benchmark_data=sentences)

    benchmark_result = benchmark_runner.benchmark_function(measure_function, batch_size=50)

    assert isinstance(benchmark_result, float)
