from testcontainers.core.image import DockerImage
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy
import pytest
import httpx

OLLAMA_PORT = 11434

# Model is always quantized, default embeddinggemma:300m is -bf16
@pytest.mark.parametrize('quantization', ['-bf16', '-qat-q8_0', '-qat-q4_0'])
def test_build_ollama(quantization, sentences):
    wait_strategy = HttpWaitStrategy(OLLAMA_PORT, '/api/tags').with_method('GET')
    with DockerImage(path="building/ollama",
                     dockerfile_path='Containerfile',
                     tag="ollama_container:test",
                     buildargs = {"QUANTIZATION": quantization}
                     ) as image:
        with DockerContainer(str(image)).with_exposed_ports(OLLAMA_PORT) as ollama_container:
            ollama_container.waiting_for(wait_strategy)

            port = ollama_container.get_exposed_port(OLLAMA_PORT)

            model_name = f"embeddinggemma:300m{quantization}"
            url_ollama = f"http://localhost:{port}/api/embed"

            payload = {
                "model": model_name,
                "input": sentences,
            }

            response = httpx.post(url_ollama, json=payload, timeout=60)
            assert response.status_code == 200, 'Invalid ollama embedding response!'
