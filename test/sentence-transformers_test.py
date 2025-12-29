import subprocess

from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy
import pytest
import httpx

from consts import SENTENCE_TRANSFORMERS_PORT as ST_PORT

@pytest.mark.parametrize('quantization', ['test'])
def test_build_st(quantization, sentences):
    image_name = 'sentence_transformers_embedding:test'
    building_command = ['podman',
                   'image',
                   'build',
                   f'--tag={image_name}',
                   '-f',
                   './building/sentence-transformers/Containerfile',
                   '--secret',
                   'id=hf_token,src=./secrets/hf_token.txt',
                   '.']

    if quantization:
        building_command.extend(['--build-arg',
                   f'QUANTIZATION={quantization}'])

    build_result = subprocess.run(building_command)
    assert build_result.returncode == 0, 'Building image failed'

    wait_strategy = HttpWaitStrategy(ST_PORT, '/health').with_method('GET')
    with DockerContainer(image_name).with_exposed_ports(ST_PORT).waiting_for(wait_strategy) as st_container:
        port = st_container.get_exposed_port(ST_PORT)
        url_st = f"http://localhost:{port}/api/embed"

        response = httpx.post(url_st, json=sentences, timeout=60)
        assert response.status_code == 200, 'Invalid onnx embedding response!'
