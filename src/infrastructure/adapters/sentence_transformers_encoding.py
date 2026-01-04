from sentence_transformers import SentenceTransformer

from src.infrastructure.utils.paths import get_model_root_path

from src.share.settings.quantization import Quantization, quantization_settings

model_root = get_model_root_path()

match quantization_settings.QUANTIZATION:
    case Quantization.INT4:
        quantization = "int4"
    case Quantization.INT8:
        quantization = "int8"
    case None:
        quantization = ""
    case _:
        raise Exception(
            f"Invalid quantization option {quantization_settings.QUANTIZATION}"
        )

model_path = f"{model_root}/sentence_transformers/embeddinggemma-300m{quantization}"

model = SentenceTransformer(model_path, device="cpu")


def encode(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
