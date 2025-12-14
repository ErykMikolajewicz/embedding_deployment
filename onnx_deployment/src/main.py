import uvicorn
from fastapi import FastAPI

from src.models import EmbedRequest, EmbedResponse
from src.services import encode

app = FastAPI(
    title="EmbeddingGemma ONNX Service",
    version="1.0.0",
)


@app.post("/api/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    embeddings = encode(request.input)
    embeddings = EmbedResponse(embeddings=embeddings)
    return embeddings


if __name__ == "__main__":
    uvicorn.run(app, port=11434)
