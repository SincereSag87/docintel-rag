from app.evaluation.models import RetrievalEvaluationMetrics
from app.retrieval.models import RetrievalResult


def evaluate_retrieval(
    results: list[RetrievalResult],
    expected_documents: list[str],
) -> RetrievalEvaluationMetrics:
    retrieved_documents = [result.filename for result in results]
    distances = [result.distance for result in results]

    if not expected_documents:
        return RetrievalEvaluationMetrics(
            hit=not retrieved_documents,
            recall=1.0 if not retrieved_documents else 0.0,
            reciprocal_rank=1.0 if not retrieved_documents else 0.0,
            expected_document_coverage=1.0 if not retrieved_documents else 0.0,
            retrieved_documents=retrieved_documents,
            distances=distances,
            expected_ranks={},
        )

    expected_set = set(expected_documents)
    retrieved_expected = expected_set.intersection(retrieved_documents)
    expected_ranks: dict[str, int] = {}
    first_rank: int | None = None

    for rank, filename in enumerate(retrieved_documents, start=1):
        if filename in expected_set and filename not in expected_ranks:
            expected_ranks[filename] = rank
            first_rank = rank if first_rank is None else min(first_rank, rank)

    recall = len(retrieved_expected) / len(expected_set)
    return RetrievalEvaluationMetrics(
        hit=bool(retrieved_expected),
        recall=recall,
        reciprocal_rank=0.0 if first_rank is None else 1 / first_rank,
        expected_document_coverage=recall,
        retrieved_documents=retrieved_documents,
        distances=distances,
        expected_ranks=expected_ranks,
    )
