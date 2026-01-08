from sentence_transformers import SentenceTransformer

from src.domain.quantization import Quantization
from src.share.consts import MODEL_ROOT

class SentenceTransformersEncoder:
    def __init__(self, quantization: str):
        self.__initialize_model(quantization)

    def __initialize_model(self, quantization: str):
        model_path = f"{MODEL_ROOT}/sentence_transformers/embeddinggemma-300m"
        self.__model = SentenceTransformer(model_path, device="cpu")

        match quantization:
            case None:
                pass
            case Quantization.BF16:
                self.__model = self.__model.bfloat16()
            case _:
                raise Exception(f"Invalid quantization option {quantization}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.__model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
