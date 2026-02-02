import subprocess

from testcontainers.core.container import DockerContainer
from testcontainers.core.image import DockerImage
from testcontainers.core.wait_strategies import HttpWaitStrategy

from src_bench.consts import OLLAMA_DEFAULT_PORT


class OnnxContainerInstantiate:
    def __init__(self, port: int, quantization: str):
        self.__image = DockerImage(
            path=".",
            dockerfile_path="building/onnx/Containerfile",
            tag="onnx_embedding:benchmark",
            buildargs={"QUANTIZATION": quantization},
        )

        image_str = str(self.__image)
        wait_strategy = HttpWaitStrategy(port, "/health").with_method("GET")
        self.__container = (
            DockerContainer(image_str)
            .with_exposed_ports(port)
            .waiting_for(wait_strategy)
            .with_env("QUANTIZATION", quantization)
            .with_bind_ports(port, port)
        )

    def __enter__(self):
        self.__image.build()
        self.__container.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__container.stop()
        self.__image.remove()


class OllamaContainerInstantiate:
    def __init__(self, port: int, quantization: str):
        quantization_mapper = {"bf16": "-bf16", "int8": "-qat-q8_0", "int4": "-qat-q4_0"}
        quantization = quantization_mapper[quantization]

        self.__image = DockerImage(
            path="building/ollama",
            dockerfile_path="Containerfile",
            tag="ollama_embedding:benchmark",
            buildargs={"QUANTIZATION": quantization},
        )

        image_str = str(self.__image)
        wait_strategy = HttpWaitStrategy(OLLAMA_DEFAULT_PORT, "/api/tags").with_method("GET")
        self.__container = (
            DockerContainer(image_str)
            .with_exposed_ports(port)
            .waiting_for(wait_strategy)
            .with_env("QUANTIZATION", quantization)
            .with_bind_ports(OLLAMA_DEFAULT_PORT, port)
        )

    def __enter__(self):
        self.__image.build()
        self.__container.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__container.stop()
        self.__image.remove()


class SentenceTransformersContainerInstantiate:
    def __init__(self, port: int, quantization: str):
        wait_strategy = HttpWaitStrategy(port, "/health").with_method("GET")
        self.__container = (
            DockerContainer("sentence_transformers_embedding:benchmark")
            .with_exposed_ports(port)
            .waiting_for(wait_strategy)
            .with_env("QUANTIZATION", quantization)
            .with_bind_ports(port, port)
        )

    def __enter__(self):
        image_name = "sentence_transformers_embedding:benchmark"
        building_command = [
            "podman",
            "image",
            "build",
            f"--tag={image_name}",
            "-f",
            "./building/sentence_transformers/Containerfile",
            "--secret",
            "id=hf_token,src=./secrets/hf_token.txt",
            ".",
        ]

        subprocess.run(building_command)
        self.__container.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__container.stop()
