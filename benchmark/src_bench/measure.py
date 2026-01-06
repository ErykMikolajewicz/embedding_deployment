from datetime import datetime, timedelta

from src.embeddings import get_embeddings

def rest_test(articles: list[str], batch_size: int) -> timedelta:
    num_articles = len(articles)
    t1 = datetime.now()
    for i in range(0, num_articles, batch_size):
        batch_articles = articles[i:i + batch_size]
        get_embeddings(batch_articles)

    t2 = datetime.now()
    return t2 - t1