import requests

from src.settings import benchmark_settings

OLLAMA_HOST = benchmark_settings.HOST
MODEL_NAME = f"embeddinggemma:300m{benchmark_settings.QUANTIZATION}"

def get_embeddings(texts: list[str]) -> list[list[float]]:

    url = f"{OLLAMA_HOST}/api/embed"

    payload = {
        "model": MODEL_NAME,
        "input": texts,
    }

    resp = requests.post(url, json=payload, timeout=60)

    data = resp.json()

    embeddings = data['embeddings']

    return embeddings
