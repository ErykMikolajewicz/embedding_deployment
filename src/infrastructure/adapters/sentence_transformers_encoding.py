from sentence_transformers import SentenceTransformer

from src.domain.enums import Quantization
from src.infrastructure.utils import get_model_root_path


class SentenceTransformersEncoder:
    def __init__(self, quantization: str):
        self.__initialize_model(quantization)

    def __initialize_model(self, quantization: str):
        model_root = get_model_root_path()
        model_path = f"{model_root}/sentence_transformers/embeddinggemma-300m"
        model = SentenceTransformer(model_path, device="cpu")

        match quantization:
            case None:
                self.__model = model
            case Quantization.BF16:
                self.__model = model.bfloat16()
            case _:
                raise Exception(f"Invalid quantization option {quantization}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.__model.encode(
            texts,
            batch_size=50,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
