from enum import StrEnum


class FrameworkType(StrEnum):
    ONNX = "ONNX"
    OLLAMA = "OLLAMA"
    SENTENCE_TRANSFORMERS = "SENTENCE_TRANSFORMERS"


class AdapterType(StrEnum):
    DIRECT = "DIRECT"
    REST = "REST"
