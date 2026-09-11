import re

from app.documents.models import Citation
from app.evaluation.models import AnswerEvaluationMetrics, EvaluationCase
from app.rag.prompts import INSUFFICIENT_INFORMATION_ANSWER


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def expected_phrase_coverage(answer: str, expected_phrases: list[str]) -> float:
    if not expected_phrases:
        return 1.0
    normalized_answer = normalize_for_match(answer)
    matched = sum(
        1
        for phrase in expected_phrases
        if normalize_for_match(phrase) in normalized_answer
    )
    return matched / len(expected_phrases)


def citation_precision_recall(
    citations: list[Citation],
    expected_documents: list[str],
) -> tuple[float, float]:
    cited = [citation.filename for citation in citations]
    if not cited and not expected_documents:
        return 1.0, 1.0
    if not cited:
        return 0.0, 0.0 if expected_documents else 1.0

    expected_set = set(expected_documents)
    cited_set = set(cited)
    true_positive = len(cited_set.intersection(expected_set))
    precision = true_positive / len(cited_set)
    recall = 1.0 if not expected_set else true_positive / len(expected_set)
    return precision, recall


def evaluate_answer(
    case: EvaluationCase,
    answer: str,
    answered: bool,
    citations: list[Citation],
) -> AnswerEvaluationMetrics:
    phrase_coverage = expected_phrase_coverage(answer, case.expected_answer_contains)
    precision, recall = citation_precision_recall(citations, case.expected_documents)

    if case.answerable:
        answer_correct = answered and phrase_coverage == 1.0
        citation_correct = bool(citations) and recall == 1.0 and precision > 0
        unknown_behavior_correct = None
    else:
        insufficient = normalize_for_match(answer) == normalize_for_match(
            INSUFFICIENT_INFORMATION_ANSWER
        )
        unknown_behavior_correct = not answered and insufficient and not citations
        answer_correct = unknown_behavior_correct
        citation_correct = not citations

    return AnswerEvaluationMetrics(
        answer_correct=answer_correct,
        expected_phrase_coverage=phrase_coverage,
        citation_precision=precision,
        citation_recall=recall,
        citation_correct=citation_correct,
        unknown_behavior_correct=unknown_behavior_correct,
    )
