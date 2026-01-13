from unittest.mock import patch

from src_bench.domain.models import FrameworkResult
from src_bench.main import run_benchmark


def test_run_benchmark():
    with patch("src_bench.domain.services.general.CONFIG_FILE_NAME", "config.json.example"):
        benchmark_result = run_benchmark()

    assert len(benchmark_result) > 0

    for framework_result in benchmark_result:
        assert isinstance(framework_result, FrameworkResult)
