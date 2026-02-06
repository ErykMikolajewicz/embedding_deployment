from enum import Enum, StrEnum


class FrameworkType(StrEnum):
    ONNX = "ONNX"
    OLLAMA = "OLLAMA"
    SENTENCE_TRANSFORMERS = "SENTENCE_TRANSFORMERS"


class AdapterType(StrEnum):
    DIRECT = "DIRECT"
    REST = "REST"


class Quantization(Enum):
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"
    NONE = None
