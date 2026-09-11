from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.documents.models import Citation
from app.retrieval.models import RetrievalResult


class RetrievedContext(BaseModel):
    query: str
    results: list[RetrievalResult]
    formatted_context: str
    total_characters: int
    used_result_count: int

    model_config = ConfigDict(frozen=True)


class ParsedRAGResponse(BaseModel):
    answer: str = Field(min_length=1)
    answered: bool
    source_numbers: list[int] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)

    @field_validator("source_numbers")
    @classmethod
    def source_numbers_must_be_positive(cls, value: list[int]) -> list[int]:
        if any(number < 1 for number in value):
            raise ValueError("Source numbers must be positive 1-based indices.")
        return value


class GroundedAnswer(BaseModel):
    question: str
    answer: str
    answered: bool
    citations: list[Citation]
    model: str
    retrieval_count: int
    confidence_label: str | None = None
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)
