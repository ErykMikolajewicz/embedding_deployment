from enum import StrEnum


class DeploymentOption(StrEnum):
    OLLAMA = "ollama"
    ONNX = "onnx"


class QuantizationType(StrEnum):
    INT4 = 'int4'
    INT8 = 'int8'
    FP16 = 'fp16'
    BF16 = 'bf16'