from src.domain.protocols import TextsEncoder
from src.infrastructure.enums import DecoderType
from src.share.settings.app import app_settings
from src.share.settings.quantization import quantization_settings

encoder = None

def initialize_encoder():
    global encoder
    match app_settings.DECODER_TYPE:
        case DecoderType.ONNX:
            from src.infrastructure.adapters.onnx_encoding import OnnxEncoder

            encoder = OnnxEncoder(quantization_settings.QUANTIZATION)
        case DecoderType.SENTENCE_TRANSFORMERS:
            from src.infrastructure.adapters.sentence_transformers_encoding import SentenceTransformersEncoder

            encoder = SentenceTransformersEncoder(quantization_settings.QUANTIZATION)
        case _:
            raise Exception("Invalid encoder type!")


def get_texts_encoder() -> TextsEncoder:
    return encoder.encode
