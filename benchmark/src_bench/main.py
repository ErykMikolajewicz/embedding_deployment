import os

from src_bench.consts import DEFAULT_HTTP_PORT
from src_bench.domain.models import FrameworkResult
from src_bench.domain.services.benchmarking import BenchmarkRunner
from src_bench.domain.services.general import get_benchmark_config, get_benchmark_data
from src_bench.infrastructure.adapters import get_adapter
from src_bench.infrastructure.containers import get_environment_preparator


def run_benchmark() -> list[FrameworkResult]:
    os.environ["ENVIRONMENT"] = "LOCAL"

    benchmark_config = get_benchmark_config()
    benchmark_data = get_benchmark_data()

    measure_number = benchmark_config.measure_number
    adapter_type = benchmark_config.adapter_type

    results = []
    for framework_config in benchmark_config.frameworks_config:
        framework_type = framework_config.framework
        for quantization in framework_config.quantization_types:
            environment_preparator_class = get_environment_preparator(framework_type, adapter_type)
            environment_preparator = environment_preparator_class(quantization)
            adapter_class = get_adapter(framework_type, adapter_type)
            adapter = adapter_class(DEFAULT_HTTP_PORT, quantization)

            measure_function = adapter.get_embeddings

            benchmark_runner = BenchmarkRunner(measure_number, benchmark_data)

            with environment_preparator:
                for batch_size in framework_config.batches_sizes:
                    adapter_result = benchmark_runner.benchmark_function(measure_function, batch_size)

                    framework_result = FrameworkResult(framework_type, batch_size, quantization, adapter_result)

                    results.append(framework_result)

                    print(framework_result)

    return results
