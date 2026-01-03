from sentence_transformers import SentenceTransformer

from src.infrastructure.utils.paths import get_model_root_path

model_root = get_model_root_path()

model_path = f"{model_root}/sentence_transformers/embeddinggemma-300m"

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
