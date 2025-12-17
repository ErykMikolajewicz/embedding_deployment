from pathlib import Path

from src.extraction import extract_articles, extract_text
from src.measure import rest_test

act_id, act_name = 'U-2024-1-572', 'kodeks_administracyjny'

act_path = Path('data') / (act_name + '.pdf')
act_text = extract_text(act_path)
articles = extract_articles(act_text, act_id)

article_texts = [article['text'] for article in articles]
article_texts: list[str] = sorted(article_texts, key=len)

measure_numbers = 3
batches = (5, 10, 20, 50)
for batch_size in batches:
    measure_results = []
    for measure_number in range(1, measure_numbers + 1):
        print(measure_number)
        result = rest_test(article_texts, batch_size)
        result_number = result.seconds + result.microseconds/1_000_000
        result = round(result_number, 2)
        measure_results.append(result)
    min_result = min(measure_results)
    print(f'For batch size {batch_size} minimum time: {min_result}')


