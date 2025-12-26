import uvicorn
from fastapi import FastAPI, Depends

from src.framework.models import Texts, Embeddings
from src.framework.dependencies import get_texts_encoder
from src.domain.protocols import TextsEncoder

app = FastAPI(
    title="EmbeddingGemma Service"
)


@app.post("/api/embed", response_model=Embeddings)
def embed(texts: Texts, texts_encoder: TextsEncoder = Depends(get_texts_encoder)):
    texts = texts.model_dump()
    embeddings = texts_encoder(texts)
    embeddings = Embeddings.model_validate(embeddings)
    return embeddings

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, port=11434)
