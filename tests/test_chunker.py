from app.chunking.recursive_chunker import RecursiveTextChunker
from app.documents.models import Document


def make_document(text: str) -> Document:
    return Document(
        id="sha256:test",
        filename="handbook.txt",
        text=text,
        metadata={"department": "people", "page_count": 2},
    )


def test_chunker_splits_on_paragraph_boundaries() -> None:
    document = make_document("PTO policy.\n\nExpense policy.\n\nRemote work policy.")
    chunks = RecursiveTextChunker(chunk_size=25, chunk_overlap=0).chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "PTO policy.",
        "Expense policy.",
        "Remote work policy.",
    ]


def test_chunker_splits_long_document() -> None:
    document = make_document(" ".join(f"word{i}" for i in range(100)))
    chunks = RecursiveTextChunker(chunk_size=80, chunk_overlap=0).chunk(document)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 80 for chunk in chunks)


def test_chunker_applies_overlap() -> None:
    document = make_document("Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.")
    chunks = RecursiveTextChunker(chunk_size=25, chunk_overlap=5).chunk(document)

    assert chunks[1].text.startswith("raph.")


def test_chunker_stable_ordering_and_deterministic_ids() -> None:
    document = make_document("A first paragraph.\n\nA second paragraph.")
    chunker = RecursiveTextChunker(chunk_size=25, chunk_overlap=3)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_index for chunk in first] == [0, 1]
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_chunker_produces_no_empty_chunks() -> None:
    document = make_document("Alpha.\n\n\n\nBeta.")
    chunks = RecursiveTextChunker(chunk_size=10, chunk_overlap=0).chunk(document)

    assert all(chunk.text.strip() for chunk in chunks)


def test_chunker_metadata_includes_document_fields_and_page_marker() -> None:
    document = make_document("[Page 2]\nSecurity policy requires MFA.")
    chunks = RecursiveTextChunker(chunk_size=800, chunk_overlap=0).chunk(document)

    metadata = chunks[0].metadata
    assert metadata["filename"] == "handbook.txt"
    assert metadata["source_type"] == "file"
    assert metadata["document_id"] == "sha256:test"
    assert metadata["chunk_index"] == 0
    assert metadata["page"] == 2
