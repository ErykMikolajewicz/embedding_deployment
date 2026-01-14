import os

from src_bench.consts import DEFAULT_HTTP_PORT
from src_bench.domain.enums import AdapterType
from src_bench.domain.models import FrameworkResult
from src_bench.domain.services.benchmarking import BenchmarkRunner
from src_bench.domain.services.general import get_benchmark_config, get_benchmark_data
from src_bench.infrastructure.adapters import get_adapter_rest, get_direct_adapter
from src_bench.infrastructure.containers import get_container_instantiate


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
            match adapter_type:
                case AdapterType.REST:
                    container_instantiate_class = get_container_instantiate(framework_type)
                    container_instantiate = container_instantiate_class(quantization)
                    adapter_class = get_adapter_rest(framework_type)
                    adapter = adapter_class(DEFAULT_HTTP_PORT, quantization)
                case AdapterType.DIRECT:
                    adapter_class = get_direct_adapter(framework_type)
                    adapter = adapter_class(quantization)
                case _:
                    raise Exception(f"Invalid adapter type {adapter_type} !")

            measure_function = adapter.get_embeddings

            benchmark_runner = BenchmarkRunner(measure_number, benchmark_data)

            for batch_size in framework_config.batches_sizes:
                if adapter_type == AdapterType.DIRECT:
                    adapter_result = benchmark_runner.benchmark_function(measure_function, batch_size)
                else:
                    with container_instantiate:
                        adapter_result = benchmark_runner.benchmark_function(measure_function, batch_size)

                framework_result = FrameworkResult(framework_type, batch_size, quantization, adapter_result)

                results.append(framework_result)

                print(framework_result)

    return results
