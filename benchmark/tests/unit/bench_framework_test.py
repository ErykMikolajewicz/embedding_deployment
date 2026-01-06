from unittest.mock import patch
from datetime import timedelta

import pytest

from src_bench.domain.enums import FrameworkType
from src_bench.domain.models import FrameworkBenchConfig, FrameworkResult
from src_bench.domain.services.benchmarking import benchmark_framework


@pytest.mark.parametrize("framework_type", FrameworkType)
def test_benchmark_framework(framework_type: FrameworkType, sentences):
    batches_sizes = [5, 10, 20, 50]
    quantization_types = ["int4", "int8", "fp16"]
    framework_config = FrameworkBenchConfig(
        framework=framework_type, batches_sizes=batches_sizes, quantization_types=quantization_types
    )
    return_value = timedelta(seconds=1)
    with patch("src_bench.domain.services.benchmarking.rest_test", return_value=return_value):
        benchmark_results = benchmark_framework(framework_config, sentences)

    excepted_results_number = len(batches_sizes) * len(quantization_types)
    assert len(benchmark_results) == excepted_results_number

    for result in benchmark_results:
        assert isinstance(result, FrameworkResult)
