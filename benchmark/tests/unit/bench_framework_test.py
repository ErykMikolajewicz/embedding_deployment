from unittest.mock import Mock

from src_bench.domain.models import BenchRunData
from src_bench.domain.services.benchmarking import benchmark_function


def test_benchmark_framework(sentences):
    measure_function = Mock(return_value=None)
    bench_run_data = BenchRunData(measure_number=5, measure_function=measure_function, data=sentences, batch_size=50)

    benchmark_result = benchmark_function(bench_run_data)

    assert isinstance(benchmark_result, float)
