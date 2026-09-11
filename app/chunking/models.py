from pydantic import BaseModel, ConfigDict, Field


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    model_config = ConfigDict(frozen=True)
