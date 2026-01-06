from src_bench.domain.models import FrameworkResult
from src_bench.domain.services.benchmarking import benchmark_framework
from src_bench.domain.services.general import get_benchmark_config, get_benchmark_data


def run_benchmark() -> list[FrameworkResult]:
    benchmark_config = get_benchmark_config()
    benchmark_data = get_benchmark_data()

    results = []
    for framework_config in benchmark_config:
        framework_results = benchmark_framework(framework_config, benchmark_data)
        results.extend(framework_results)

    return results
