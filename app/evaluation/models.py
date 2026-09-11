from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvaluationDocument(BaseModel):
    filename: str = Field(min_length=1)
    content: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerable: bool
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_documents: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    top_k: int | None = Field(default=None, gt=0)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_expectations(self) -> "EvaluationCase":
        if self.answerable and not self.expected_answer_contains:
            raise ValueError("Answerable cases require expected_answer_contains.")
        if self.answerable and not self.expected_documents:
            raise ValueError("Answerable cases require expected_documents.")
        return self


class RAGEvaluationDataset(BaseModel):
    name: str = Field(min_length=1)
    documents: list[EvaluationDocument] = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    model_config = ConfigDict(frozen=True)

    @field_validator("cases")
    @classmethod
    def case_ids_must_be_unique(cls, cases: list[EvaluationCase]) -> list[EvaluationCase]:
        case_ids = [case.id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case IDs must be unique.")
        return cases

    @model_validator(mode="after")
    def expected_documents_must_exist(self) -> "RAGEvaluationDataset":
        filenames = {document.filename for document in self.documents}
        missing = sorted(
            {
                expected
                for case in self.cases
                for expected in case.expected_documents
                if expected not in filenames
            }
        )
        if missing:
            raise ValueError(f"Expected documents are missing from dataset: {missing}.")
        return self


class RetrievalEvaluationMetrics(BaseModel):
    hit: bool
    recall: float
    reciprocal_rank: float
    expected_document_coverage: float
    retrieved_documents: list[str]
    distances: list[float]
    expected_ranks: dict[str, int]

    model_config = ConfigDict(frozen=True)


class AnswerEvaluationMetrics(BaseModel):
    answer_correct: bool
    expected_phrase_coverage: float
    citation_precision: float
    citation_recall: float
    citation_correct: bool
    unknown_behavior_correct: bool | None = None

    model_config = ConfigDict(frozen=True)


class RAGEvaluationCaseResult(BaseModel):
    case_id: str
    question: str
    model: str
    success: bool
    answerable: bool
    answered: bool | None = None
    answer: str = ""
    expected_documents: list[str]
    retrieved_documents: list[str] = Field(default_factory=list)
    cited_documents: list[str] = Field(default_factory=list)
    retrieval_hit: bool = False
    retrieval_recall: float = 0.0
    reciprocal_rank: float = 0.0
    expected_document_coverage: float = 0.0
    answer_correct: bool = False
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_correct: bool = False
    unknown_behavior_correct: bool | None = None
    parse_valid: bool = False
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None

    model_config = ConfigDict(frozen=True)


class RAGEvaluationReport(BaseModel):
    dataset_name: str
    model: str
    retrieval_only: bool = False
    cases_total: int
    cases_passed: int
    retrieval_hit_rate: float
    mean_retrieval_recall: float
    mrr: float
    answer_accuracy: float
    average_citation_precision: float
    average_citation_recall: float
    citation_accuracy: float
    unknown_answer_accuracy: float
    parse_success_rate: float
    average_latency_ms: float
    failed_case_count: int
    case_results: list[RAGEvaluationCaseResult]

    model_config = ConfigDict(frozen=True)
