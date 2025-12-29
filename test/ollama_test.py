from test.consts import OLLAMA_PORT

import httpx
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.image import DockerImage
from testcontainers.core.wait_strategies import HttpWaitStrategy


# Model is always quantized, default embeddinggemma:300m is -bf16
@pytest.mark.parametrize("quantization", ["-bf16", "-qat-q8_0", "-qat-q4_0"])
def test_ollama_embedding(quantization, sentences, measure_similarity):
    wait_strategy = HttpWaitStrategy(OLLAMA_PORT, "/api/tags").with_method("GET")
    with DockerImage(
        path="building/ollama",
        dockerfile_path="Containerfile",
        tag="ollama_embedding:test",
        buildargs={"QUANTIZATION": quantization},
    ) as image:
        with DockerContainer(str(image)).with_exposed_ports(
            OLLAMA_PORT
        ) as ollama_container:
            ollama_container.waiting_for(wait_strategy)

            port = ollama_container.get_exposed_port(OLLAMA_PORT)

            model_name = f"embeddinggemma:300m{quantization}"
            url_ollama = f"http://localhost:{port}/api/embed"

            payload = {
                "model": model_name,
                "input": sentences,
            }

            response = httpx.post(url_ollama, json=payload, timeout=60)
            print(response.text)

        assert response.status_code == 200, "Invalid ollama embedding response!"

        result = response.json()
        vectors = result["embeddings"]
        measure_similarity(vectors)
