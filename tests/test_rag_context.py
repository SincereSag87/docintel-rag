from app.rag.context import ContextBuilder
from app.retrieval.models import RetrievalResult


def result(index: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        filename=f"file-{index}.txt",
        text=text,
        distance=0.1 * index,
        chunk_index=index,
        metadata={"page": index, "filename": f"file-{index}.txt"},
    )


def test_context_builder_uses_stable_source_numbering() -> None:
    context = ContextBuilder(max_chars=5000).build("question", [result(1, "alpha")])

    assert "[SOURCE 1]" in context.formatted_context
    assert "Filename: file-1.txt" in context.formatted_context
    assert "Distance: 0.100000" in context.formatted_context
    assert context.used_result_count == 1


def test_context_builder_preserves_highest_ranked_results_under_limit() -> None:
    results = [result(1, "alpha" * 20), result(2, "beta" * 200)]

    context = ContextBuilder(max_chars=220).build("question", results)

    assert context.results == [results[0]]
    assert "file-1.txt" in context.formatted_context
    assert "file-2.txt" not in context.formatted_context


def test_context_builder_truncates_single_oversized_source() -> None:
    context = ContextBuilder(max_chars=120).build("question", [result(1, "x" * 1000)])

    assert context.total_characters == 120
    assert context.used_result_count == 1


def test_context_builder_includes_page_metadata() -> None:
    context = ContextBuilder(max_chars=5000).build("question", [result(2, "policy text")])

    assert "Page: 2" in context.formatted_context
