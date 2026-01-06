import json
from pathlib import Path

from src_bench.consts import CONFIG_FILE_NAME
from src_bench.domain.models import FrameworkBenchConfig
from src_bench.domain.services.extraction import extract_articles, extract_text


def get_benchmark_config() -> list[FrameworkBenchConfig]:
    with open(CONFIG_FILE_NAME, "r", encoding="utf-8") as config_file:
        benchmark_config = json.load(config_file)

    return [FrameworkBenchConfig(**framework_config) for framework_config in benchmark_config]


def get_benchmark_data():
    act_path = Path("benchmark") / "data" / "kodeks_administracyjny.pdf"
    act_text = extract_text(act_path)
    articles = extract_articles(act_text)

    articles = sorted(articles, key=len)

    return tuple(articles)
