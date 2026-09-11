from pydantic import BaseModel, ConfigDict, Field


class EmbeddingResponse(BaseModel):
    model: str
    embedding: list[float] = Field(min_length=1)

    model_config = ConfigDict(frozen=True)
