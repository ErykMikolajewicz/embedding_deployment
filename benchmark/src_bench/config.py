import json
from dataclasses import dataclass
from pathlib import Path

from src.domain.quantization import Quantization
from src_bench.consts import CONFIG_FILE_NAME
from src_bench.domain.enums import AdapterType, FrameworkType


@dataclass
class FrameworkBenchConfig:
    framework: FrameworkType
    batch_sizes: list[int]
    quantization_types: list[Quantization]


@dataclass
class BenchConfig:
    measure_number: int
    adapter_type: AdapterType
    rest_port: int | None
    frameworks_config: list[FrameworkBenchConfig]


def get_benchmark_config() -> BenchConfig:
    config_path = Path("benchmark") / CONFIG_FILE_NAME
    with open(config_path, "r", encoding="utf-8") as config_file:
        benchmark_config_json = json.load(config_file)

    frameworks_config = []
    for framework_config in benchmark_config_json["frameworks_config"]:
        framework_config = FrameworkBenchConfig(**framework_config)
        frameworks_config.append(framework_config)
    benchmark_config_json["frameworks_config"] = frameworks_config

    benchmark_config = BenchConfig(**benchmark_config_json)

    return benchmark_config
