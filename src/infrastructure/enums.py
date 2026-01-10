from enum import StrEnum


class DecoderType(StrEnum):
    ONNX = "ONNX"
    SENTENCE_TRANSFORMERS = "SENTENCE_TRANSFORMERS"


class Environment(StrEnum):
    LOCAL = "LOCAL"
    CONTAINER = "CONTAINER"
