from app.evaluation.retrieval_metrics import evaluate_retrieval
from app.retrieval.models import RetrievalResult


def result(filename: str, distance: float = 0.1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{filename}:chunk",
        document_id=f"{filename}:doc",
        filename=filename,
        text="text",
        distance=distance,
        chunk_index=0,
        metadata={},
    )


def test_retrieval_hit_recall_and_mrr() -> None:
    metrics = evaluate_retrieval(
        [result("wrong.txt"), result("policy.txt")],
        ["policy.txt"],
    )

    assert metrics.hit is True
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.expected_ranks == {"policy.txt": 2}


def test_retrieval_multi_source_coverage() -> None:
    metrics = evaluate_retrieval(
        [result("handbook.txt"), result("other.txt")],
        ["handbook.txt", "expense.txt"],
    )

    assert metrics.hit is True
    assert metrics.recall == 0.5
    assert metrics.expected_document_coverage == 0.5


def test_retrieval_no_relevant_result() -> None:
    metrics = evaluate_retrieval([result("wrong.txt")], ["policy.txt"])

    assert metrics.hit is False
    assert metrics.recall == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_retrieval_unanswerable_with_no_results() -> None:
    metrics = evaluate_retrieval([], [])

    assert metrics.hit is True
    assert metrics.recall == 1.0
