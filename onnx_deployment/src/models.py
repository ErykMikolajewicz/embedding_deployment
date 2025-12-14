from pydantic import BaseModel

class EmbedRequest(BaseModel):
    model: str
    input: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
