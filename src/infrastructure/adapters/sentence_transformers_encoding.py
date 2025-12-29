from sentence_transformers import SentenceTransformer

MODEL_ID_OR_PATH = "/embedding_deployment/models/embeddinggemma-300m"

model = SentenceTransformer(MODEL_ID_OR_PATH, device="cpu")

def encode(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings.tolist()
