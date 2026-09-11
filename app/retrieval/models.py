from pydantic import BaseModel, ConfigDict


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    distance: float
    chunk_index: int
    metadata: dict[str, object]

    model_config = ConfigDict(frozen=True)
