from pydantic import BaseModel, ConfigDict


class VectorStoreStats(BaseModel):
    collection: str
    count: int
    path: str | None = None

    model_config = ConfigDict(frozen=True)
