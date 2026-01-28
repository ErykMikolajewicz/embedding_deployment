
from src.domain.types import EncodeTexts
from src.infrastructure.enums import DecoderType
from src.share.settings.app import AppSettings
from src.share.settings.quantization import quantization_settings


def initialize_encoder() -> EncodeTexts:
    app_settings = AppSettings()
    match app_settings.DECODER_TYPE:
        case DecoderType.ONNX:
            from src.infrastructure.adapters.onnx_encoding import OnnxEncoder

            encoder = OnnxEncoder(quantization_settings.QUANTIZATION)
        case DecoderType.SENTENCE_TRANSFORMERS:
            from src.infrastructure.adapters.sentence_transformers_encoding import SentenceTransformersEncoder

            encoder = SentenceTransformersEncoder(quantization_settings.QUANTIZATION)
        case _:
            raise Exception("Invalid encoder type!")
    return encoder.encode

