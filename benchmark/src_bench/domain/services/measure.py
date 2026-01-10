from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta


def measure_execution_time(
    texts: Sequence[str], batch_size: int, get_embeddings: Callable[[Iterable[str]], list[list[float]]]
) -> timedelta:
    num_articles = len(texts)
    t1 = datetime.now()
    for i in range(0, num_articles, batch_size):
        batch_articles = texts[i : i + batch_size]
        get_embeddings(batch_articles)

    t2 = datetime.now()
    return t2 - t1
