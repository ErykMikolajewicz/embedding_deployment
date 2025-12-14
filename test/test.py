import requests
import numpy as np


OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "embeddinggemma:300m-qat-q8_0"
url_ollama = f"{OLLAMA_HOST}/api/embed"

ONNX_HOST = "http://localhost:8000"
url_onnx = f"{ONNX_HOST}/api/embed"

sentences = ["Ala ma kota.", "Primus inter pares.", "vtefwTFGIFOUEPJIebyorwqpg9f3hurp[[qf3"]


payload = {
    "model": MODEL_NAME,
    "input": sentences,
}

resp = requests.post(url_ollama, json=payload, timeout=60)
data = resp.json()
embeddings_llama = data.get("embeddings")

resp = requests.post(url_ollama, json=payload, timeout=60)

data = resp.json()

embeddings_onnx = data.get("embeddings")


def cosine_similarity(a, b):

    for v1, v2 in zip(a, b):
        v1 = np.array(v1)
        v2 = np.array(v2)

        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        result = dot / (norm_v1 * norm_v2)
        print(result)


cosine_similarity(embeddings_llama, embeddings_onnx)
