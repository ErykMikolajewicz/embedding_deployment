import json
from pathlib import Path

from src_bench.consts import CONFIG_FILE_NAME
from src_bench.domain.models import BenchConfig, FrameworkBenchConfig
from src_bench.domain.services.extraction import extract_articles, extract_text


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


def get_benchmark_data():
    act_path = Path("benchmark") / "data" / "kodeks_administracyjny.pdf"
    act_text = extract_text(act_path)
    articles = extract_articles(act_text)

    articles = sorted(articles, key=len)

    return tuple(articles)
