from app.documents.models import Citation
from app.evaluation.answer_metrics import (
    citation_precision_recall,
    evaluate_answer,
    expected_phrase_coverage,
)
from app.evaluation.models import EvaluationCase
from app.rag.prompts import INSUFFICIENT_INFORMATION_ANSWER


def citation(filename: str) -> Citation:
    return Citation(
        document_id=f"{filename}:doc",
        filename=filename,
        chunk_id=f"{filename}:chunk",
        chunk_index=0,
        excerpt="excerpt",
    )


def test_expected_phrase_matching_is_normalized() -> None:
    assert expected_phrase_coverage("Employees receive 15 DAYS.", ["15 days"]) == 1.0


def test_citation_precision_and_recall() -> None:
    precision, recall = citation_precision_recall(
        [citation("handbook.txt"), citation("wrong.txt")],
        ["handbook.txt", "expense.txt"],
    )

    assert precision == 0.5
    assert recall == 0.5


def test_answerable_case_metrics() -> None:
    case = EvaluationCase(
        id="pto",
        question="PTO?",
        answerable=True,
        expected_answer_contains=["15 days"],
        expected_documents=["handbook.txt"],
    )

    metrics = evaluate_answer(
        case,
        "Employees receive 15 days.",
        True,
        [citation("handbook.txt")],
    )

    assert metrics.answer_correct is True
    assert metrics.citation_correct is True


def test_unanswerable_case_metrics() -> None:
    case = EvaluationCase(
        id="unknown",
        question="Insurance?",
        answerable=False,
    )

    metrics = evaluate_answer(case, INSUFFICIENT_INFORMATION_ANSWER, False, [])

    assert metrics.answer_correct is True
    assert metrics.unknown_behavior_correct is True
