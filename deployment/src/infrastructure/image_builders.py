from typing import Optional
import subprocess

from deployment.src.domain.enums import QuantizationType, DeploymentOption
from deployment.src.domain.protocols import ImageBuilder


class OnnxImageBuilder:
    __quantization_mapper = {QuantizationType.INT4: "int4",
                             QuantizationType.INT8: "int8",
                             QuantizationType.FP16: "fp16"}

    @classmethod
    def build_image(cls, quantization: Optional[QuantizationType]):
        command = [
            "podman",
            "build",
            "-t",
            "onnx_gemma",
            "./onnx_deployment",
        ]

        if quantization is not None:
            quantization = cls.__quantization_mapper[quantization]
            command.extend(["--build-arg", f"QUANTIZATION={quantization}"])

        subprocess.run(command)


class OllamaImageBuilder:
    __quantization_mapper = {QuantizationType.INT4: "-qat-q4_0",
                             QuantizationType.INT8: "-qat-q8_0",
                             QuantizationType.BF16: "-bf16"}

    @classmethod
    def build_image(cls, quantization: Optional[QuantizationType]):
        command = [
            "podman",
            "build",
            "-t",
            "ollama_gemma",
            "./ollama_deployment",
        ]

        if quantization is not None:
            quantization = cls.__quantization_mapper[quantization]
            command.extend(["--build-arg", f"QUANTIZATION={quantization}"])

        subprocess.run(command)


def get_image_builder(deployment_option: DeploymentOption) -> ImageBuilder:
    match deployment_option:
        case DeploymentOption.ONNX:
            return OnnxImageBuilder()
        case DeploymentOption.OLLAMA:
            return OllamaImageBuilder()
        case _:
            raise Exception('Invalid deployment option!')
