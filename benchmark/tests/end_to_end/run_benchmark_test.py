from src_bench.domain.models import FrameworkResult
from src_bench.main import run_benchmark


def test_run_benchmark():
    benchmark_result = run_benchmark()

    assert len(benchmark_result) > 0

    for framework_results in benchmark_result:
        for framework_result in framework_results:
            assert isinstance(framework_result, FrameworkResult)
