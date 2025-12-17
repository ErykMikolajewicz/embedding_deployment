import re
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTPage


def extract_text(pdf_path: Path, top_margin=50, bottom_margin=60) -> str:
    text = ''
    for page_layout in extract_pages(pdf_path):
        if isinstance(page_layout, LTPage):
            page_height = page_layout.height
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    if element.y1 < page_height - top_margin and element.y0 > bottom_margin:
                        text += element.get_text()
    return text


def extract_articles(act_text: str, act_id: str) -> list[dict]:

    pattern = r"\n(?![ARD])"
    act_text = re.sub(pattern, " ", act_text)

    pattern = r"(\nArt\. \d+(?:[a-z])*\. )"

    articles_text = re.split(pattern, act_text)
    articles_text.pop(0) # To remove text before first article

    articles = []
    for i, article_text in enumerate(articles_text):
        if i % 2 == 0:
            continue
        article_id = act_id + articles_text[i-1]
        articles.append({'text': article_text, 'id': article_id})
    
    for i, article in enumerate(articles):
        article_text = article['text']
        if article_text.count('\n') > 0:
            new_line_index = article_text.index('\n')
            article['text'] = article_text[:new_line_index]

    articles = [article for article in articles if (re.search(r'\(uchylony\)', article['text']) is None) or len(article['text'].strip()) > 12]

    for article in articles:
        article['text'] = re.sub(r'\s+', ' ', article['text']).strip()
    
    return articles
