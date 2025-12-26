from testcontainers.core.image import DockerImage
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy
import pytest
import httpx

from consts import ONNX_PORT

# Model is always quantized, default is fp16
@pytest.mark.parametrize('quantization', ['fp16', 'int8']) # 'int4' currently not working
def test_build_onnx(quantization, sentences):
    wait_strategy = HttpWaitStrategy(ONNX_PORT, '/health').with_method('GET')
    with DockerImage(path=".",
                     dockerfile_path='building/onnx/Containerfile',
                     tag="onnx_embedding:test",
                     buildargs = {"QUANTIZATION": quantization}
                     ) as image:
        with DockerContainer(str(image)).with_exposed_ports(ONNX_PORT).waiting_for(wait_strategy) as onnx_container:
            port = onnx_container.get_exposed_port(ONNX_PORT)
            url_onnx = f"http://localhost:{port}/api/embed"

            response = httpx.post(url_onnx, json=sentences, timeout=60)
            assert response.status_code == 200, 'Invalid onnx embedding response!'