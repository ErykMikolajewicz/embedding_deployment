from src.domain.protocols import TextsEncoder
from src.infrastructure.enums import DecoderType
from src.settings import settings


def get_texts_encoder() -> TextsEncoder:
    match settings.DECODER_TYPE:
        case DecoderType.ONNX:
            from src.infrastructure.adapters.onnx_encoding import encode

            return encode
        case DecoderType.SENTENCE_TRANSFORMERS:
            from src.infrastructure.adapters.sentence_transformers_encoding import (
                encode,
            )

            return encode
        case _:
            raise Exception("Invalid encoder type!")
