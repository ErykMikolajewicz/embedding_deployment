import json

from src_bench.domain.models import FrameworkBenchConfig
from src_bench.consts import CONFIG_FILE_NAME

def get_benchmark_config() -> list[FrameworkBenchConfig]:
    with open(CONFIG_FILE_NAME, 'r', encoding='utf-8') as config_file:
        benchmark_config = json.load(config_file)

    return [FrameworkBenchConfig(**framework_config) for framework_config in benchmark_config]
