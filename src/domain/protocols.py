from typing import Protocol, Optional

from src.domain.enums import QuantizationType

class ImageBuilder(Protocol):
    __quantization_mapper: dict[QuantizationType, str]

    @classmethod
    def build_image(cls, quantization: Optional[QuantizationType]) -> None:
        ...
