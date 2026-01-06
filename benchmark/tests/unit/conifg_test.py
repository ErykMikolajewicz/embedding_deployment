from unittest.mock import patch

from src_bench.domain.services import get_benchmark_config
from src_bench.domain.models import FrameworkBenchConfig


def test_read_config():
    with patch('src_bench.domain.services.CONFIG_FILE_NAME', 'benchmark/config.json.example'):
        benchmark_config = get_benchmark_config()

    for framework_config in benchmark_config:
        assert isinstance(framework_config, FrameworkBenchConfig)