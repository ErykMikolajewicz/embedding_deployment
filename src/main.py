from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI

from src.domain.types import EncodeTexts
from src.framework.dependencies import get_texts_encoder, initialize_encoder
from src.framework.models import Embeddings, Texts


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_encoder()
    yield


app = FastAPI(title="EmbeddingGemma Service", lifespan=lifespan)


@app.post("/api/embed", response_model=Embeddings)
def embed(texts: Texts, texts_encoder: EncodeTexts = Depends(get_texts_encoder)):
    texts = texts.model_dump()
    embeddings = texts_encoder(texts)
    embeddings = Embeddings.model_validate(embeddings)
    return embeddings


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, port=11434)
