from pathlib import Path

from src_bench.domain.services.extraction import extract_articles, extract_text


def get_benchmark_data():
    act_path = Path("benchmark") / "data" / "kodeks_administracyjny.pdf"
    act_text = extract_text(act_path)
    articles = extract_articles(act_text)

    articles = sorted(articles, key=len)

    return tuple(articles)
