import subprocess

from testcontainers.core.container import DockerContainer
from testcontainers.core.image import DockerImage
from testcontainers.core.wait_strategies import HttpWaitStrategy

from src_bench.consts import DEFAULT_HTTP_PORT
from src_bench.domain.enums import AdapterType, FrameworkType
from src_bench.domain.ports import EnvironmentPreparator


def get_environment_preparator(framework_type: FrameworkType, adapter_type: AdapterType) -> type[EnvironmentPreparator]:
    match framework_type, adapter_type:
        case FrameworkType.ONNX, AdapterType.REST:
            return OnnxContainerInstantiate
        case FrameworkType.OLLAMA, AdapterType.REST:
            return OllamaContainerInstantiate
        case FrameworkType.SENTENCE_TRANSFORMERS, AdapterType.REST:
            return SentenceTransformersContainerInstantiate
        case _, AdapterType.DIRECT:
            return EmptyPreparator
        case _:
            raise Exception("Invalid adapter type!")


class EmptyPreparator:
    def __init__(self, _: str):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class OnnxContainerInstantiate:
    def __init__(self, quantization: str):
        self.__image = DockerImage(
            path=".",
            dockerfile_path="building/onnx/Containerfile",
            tag="onnx_embedding:benchmark",
            buildargs={"QUANTIZATION": quantization},
        )

        image_str = str(self.__image)
        wait_strategy = HttpWaitStrategy(DEFAULT_HTTP_PORT, "/health").with_method("GET")
        self.__container = (
            DockerContainer(image_str)
            .with_exposed_ports(DEFAULT_HTTP_PORT)
            .waiting_for(wait_strategy)
            .with_env("QUANTIZATION", quantization)
            .with_bind_ports(DEFAULT_HTTP_PORT, DEFAULT_HTTP_PORT)
        )

    def __enter__(self):
        self.__image.build()
        self.__container.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__container.stop()
        self.__image.remove()


class OllamaContainerInstantiate:
    def __init__(self, quantization: str):
        quantization_mapper = {"bf16": "-bf16", "int8": "-qat-q8_0", "int4": "-qat-q4_0"}
        quantization = quantization_mapper[quantization]

        self.__image = DockerImage(
            path="building/ollama",
            dockerfile_path="Containerfile",
            tag="ollama_embedding:benchmark",
            buildargs={"QUANTIZATION": quantization},
        )

        image_str = str(self.__image)
        wait_strategy = HttpWaitStrategy(11434, "/api/tags").with_method("GET")
        self.__container = (
            DockerContainer(image_str)
            .with_exposed_ports(DEFAULT_HTTP_PORT)
            .waiting_for(wait_strategy)
            .with_env("QUANTIZATION", quantization)
            .with_bind_ports(11434, DEFAULT_HTTP_PORT)
        )

    def __enter__(self):
        self.__image.build()
        self.__container.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__container.stop()
        self.__image.remove()


class SentenceTransformersContainerInstantiate:
    def __init__(self, quantization: str):
        wait_strategy = HttpWaitStrategy(DEFAULT_HTTP_PORT, "/health").with_method("GET")
        self.__container = (
            DockerContainer("sentence_transformers_embedding:benchmark")
            .with_exposed_ports(DEFAULT_HTTP_PORT)
            .waiting_for(wait_strategy)
            .with_env("QUANTIZATION", quantization)
            .with_bind_ports(DEFAULT_HTTP_PORT, DEFAULT_HTTP_PORT)
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
