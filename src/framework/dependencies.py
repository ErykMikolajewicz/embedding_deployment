from src.domain.protocols import TextsEncoder
from src.settings import settings
from src.infrastructure.enums import DecoderType


def get_texts_encoder() -> TextsEncoder:
    match settings.DECODER_TYPE:
        case DecoderType.ONNX:
            from src.infrastructure.adapters import encode
            return encode
        case DecoderType.SENTENCE_TRANSFORMERS:
            raise NotImplementedError
        case _:
            raise Exception('Invalid encoder type!')
