from app.documents.models import Citation
from app.evaluation.models import (
    AnswerEvaluationMetrics,
    EvaluationCase,
    EvaluationDocument,
    RAGEvaluationCaseResult,
    RAGEvaluationDataset,
    RetrievalEvaluationMetrics,
)
from app.evaluation.runner import RAGEvaluationRunner
from app.rag.models import GroundedAnswer
from app.rag.parsers import RAGParseError
from app.retrieval.models import RetrievalResult


def dataset() -> RAGEvaluationDataset:
    return RAGEvaluationDataset(
        name="unit",
        documents=[EvaluationDocument(filename="doc.txt", content="PTO is 15 days.")],
        cases=[
            EvaluationCase(
                id="pto",
                question="PTO?",
                answerable=True,
                expected_answer_contains=["15 days"],
                expected_documents=["doc.txt"],
            ),
            EvaluationCase(id="unknown", question="Insurance?", answerable=False),
        ],
    )


def retrieval_metrics(hit: bool = True) -> RetrievalEvaluationMetrics:
    return RetrievalEvaluationMetrics(
        hit=hit,
        recall=1.0 if hit else 0.0,
        reciprocal_rank=1.0 if hit else 0.0,
        expected_document_coverage=1.0 if hit else 0.0,
        retrieved_documents=["doc.txt"] if hit else [],
        distances=[0.1] if hit else [],
        expected_ranks={"doc.txt": 1} if hit else {},
    )


def answer_metrics(ok: bool = True) -> AnswerEvaluationMetrics:
    return AnswerEvaluationMetrics(
        answer_correct=ok,
        expected_phrase_coverage=1.0 if ok else 0.0,
        citation_precision=1.0 if ok else 0.0,
        citation_recall=1.0 if ok else 0.0,
        citation_correct=ok,
    )


def retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="doc.txt",
        text="PTO is 15 days.",
        distance=0.1,
        chunk_index=0,
        metadata={},
    )


class SuccessfulRAGService:
    def retrieve(self, question, top_k=None):
        return [retrieval_result()]

    def ask(self, question, model=None, top_k=None):
        return GroundedAnswer(
            question=question,
            answer="PTO is 15 days.",
            answered=True,
            citations=[
                Citation(
                    document_id="doc-1",
                    filename="doc.txt",
                    chunk_id="chunk-1",
                    chunk_index=0,
                    excerpt="PTO is 15 days.",
                )
            ],
            model=model or "llama3.2",
            retrieval_count=1,
        )


class ParseFailureRAGService(SuccessfulRAGService):
    def ask(self, question, model=None, top_k=None):
        raise RAGParseError("bad json")


class ProviderFailureRAGService(SuccessfulRAGService):
    def ask(self, question, model=None, top_k=None):
        raise RuntimeError("provider offline")


class RetrievalFailureRAGService:
    def retrieve(self, question, top_k=None):
        raise RuntimeError("retrieval failed")


def test_case_success_policy_answerable() -> None:
    assert RAGEvaluationRunner._case_success(dataset().cases[0], True, answer_metrics(True))
    assert not RAGEvaluationRunner._case_success(dataset().cases[0], False, answer_metrics(True))


def test_case_success_policy_unanswerable() -> None:
    metrics = AnswerEvaluationMetrics(
        answer_correct=True,
        expected_phrase_coverage=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        citation_correct=True,
        unknown_behavior_correct=True,
    )

    assert RAGEvaluationRunner._case_success(dataset().cases[1], False, metrics)


def test_aggregate_report() -> None:
    runner = RAGEvaluationRunner(dataset(), model="llama3.2")
    result = RAGEvaluationCaseResult(
        case_id="pto",
        question="PTO?",
        model="llama3.2",
        success=True,
        answerable=True,
        answered=True,
        answer="15 days",
        expected_documents=["doc.txt"],
        retrieved_documents=["doc.txt"],
        cited_documents=["doc.txt"],
        retrieval_hit=True,
        retrieval_recall=1.0,
        reciprocal_rank=1.0,
        answer_correct=True,
        citation_precision=1.0,
        citation_recall=1.0,
        citation_correct=True,
        parse_valid=True,
        latency_ms=100.0,
    )

    report = runner._aggregate([result])

    assert report.cases_total == 1
    assert report.cases_passed == 1
    assert report.retrieval_hit_rate == 1.0


def test_run_case_successful_case() -> None:
    runner = RAGEvaluationRunner(dataset(), model="llama3.2")

    result = runner._run_case(dataset().cases[0], SuccessfulRAGService())

    assert result.success is True
    assert result.retrieval_hit is True
    assert result.answer_correct is True


def test_run_case_records_parse_failure() -> None:
    runner = RAGEvaluationRunner(dataset(), model="llama3.2")

    result = runner._run_case(dataset().cases[0], ParseFailureRAGService())

    assert result.success is False
    assert result.parse_valid is False
    assert result.error == "bad json"


def test_run_case_records_provider_failure() -> None:
    runner = RAGEvaluationRunner(dataset(), model="llama3.2")

    result = runner._run_case(dataset().cases[0], ProviderFailureRAGService())

    assert result.success is False
    assert result.parse_valid is True
    assert result.error == "provider offline"


def test_run_case_records_retrieval_failure() -> None:
    runner = RAGEvaluationRunner(dataset(), model="llama3.2")

    result = runner._run_case(dataset().cases[0], RetrievalFailureRAGService())

    assert result.success is False
    assert result.error == "retrieval failed"


def test_run_case_retrieval_only_avoids_generation() -> None:
    runner = RAGEvaluationRunner(dataset(), model="llama3.2", retrieval_only=True)

    result = runner._run_case(dataset().cases[0], SuccessfulRAGService())

    assert result.success is True
    assert result.answer == ""
