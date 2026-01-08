from functools import partial

from src_bench.domain.enums import FrameworkType
from src_bench.domain.models import FrameworkBenchConfig, FrameworkResult
from src_bench.domain.services.measure import rest_test
from src_bench.infrastructure.adapters import get_adapter_rest


def benchmark_framework(
    framework_config: FrameworkBenchConfig, benchmark_data: tuple[str, ...]
) -> list[FrameworkResult]:
    framework = framework_config.framework
    adapter = get_adapter_rest(framework)

    measure_numbers = 3
    batches = framework_config.batches_sizes

    results = []
    for quantization in framework_config.quantization_types:
        if framework == FrameworkType.OLLAMA:
            get_embeddings = partial(adapter.get_embeddings, quantization)
        else:
            get_embeddings = adapter.get_embeddings

        for batch_size in batches:
            measure_results = []
            for measure_number in range(1, measure_numbers + 1):
                result = rest_test(benchmark_data, batch_size, get_embeddings)
                result_number = result.seconds + result.microseconds / 1_000_000
                result = round(result_number, 2)
                measure_results.append(result)
            min_result = min(measure_results)

            image_size = None

            result = FrameworkResult(framework, batch_size, quantization, min_result, image_size)
            results.append(result)

    return results
