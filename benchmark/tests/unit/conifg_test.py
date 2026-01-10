from unittest.mock import patch

from src_bench.domain.models import FrameworkBenchConfig, BenchConfig
from src_bench.domain.services.general import get_benchmark_config


def test_read_config():
    with patch("src_bench.domain.services.general.CONFIG_FILE_NAME", "config.json.example"):
        benchmark_config = get_benchmark_config()

    assert isinstance(benchmark_config, BenchConfig)

    for framework_config in benchmark_config.frameworks_config:
        assert isinstance(framework_config, FrameworkBenchConfig)
