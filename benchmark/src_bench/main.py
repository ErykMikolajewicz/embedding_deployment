from dishka import Scope

from src_bench.config import get_benchmark_config
from src_bench.domain.enums import FrameworkType, Quantization
from src_bench.domain.models import FrameworkResult
from src_bench.domain.services.measure import Measurer
from src_bench.providers import container


def run_benchmark() -> list[FrameworkResult]:
    benchmark_config = get_benchmark_config()

    results = []
    for framework_config in benchmark_config.frameworks_config:
        framework_type = framework_config.framework
        for quantization in framework_config.quantization_types:
            with container(
                context={Quantization: quantization, FrameworkType: framework_type}, scope=Scope.SESSION
            ) as session_container:
                measurer = session_container.get(Measurer)

                for batch_size in framework_config.batch_sizes:
                    result = measurer.measure_median_time(batch_size)

                    framework_result = FrameworkResult(framework_type, batch_size, quantization, result)

                    results.append(framework_result)
                    print(framework_result)

    return results


if __name__ == "__main__":
    run_benchmark()
