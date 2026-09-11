from pathlib import Path

from app.documents.models import DocumentChunk, EmbeddedChunk
from app.storage.chroma_store import ChromaVectorStore


def embedded_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    document_id: str = "doc-1",
) -> EmbeddedChunk:
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=int(chunk_id[-1]),
        metadata={
            "document_id": document_id,
            "filename": "handbook.txt",
            "source_type": "file",
            "chunk_index": int(chunk_id[-1]),
            "nested": {"skip": True},
            "none_value": None,
        },
    )
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


def test_chroma_store_adds_and_searches_chunks(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="test_docs")
    store.add_chunks(
        [
            embedded_chunk("chunk-0", "PTO is 15 days.", [1.0, 0.0]),
            embedded_chunk("chunk-1", "Expenses above $500 need approval.", [0.0, 1.0]),
        ]
    )

    results = store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk.id == "chunk-0"
    assert results[0].metadata["filename"] == "handbook.txt"
    assert isinstance(results[0].distance, float)


def test_chroma_store_upsert_prevents_duplicate_chunks(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="test_docs")
    chunk = embedded_chunk("chunk-0", "PTO is 15 days.", [1.0, 0.0])

    store.add_chunks([chunk])
    store.add_chunks([chunk])

    assert store.stats().count == 1


def test_chroma_store_deletes_document(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="test_docs")
    store.add_chunks(
        [
            embedded_chunk("chunk-0", "PTO is 15 days.", [1.0, 0.0], document_id="doc-1"),
            embedded_chunk("chunk-1", "Other document.", [0.0, 1.0], document_id="doc-2"),
        ]
    )

    store.delete_document("doc-1")

    assert store.stats().count == 1


def test_chroma_store_clear_and_stats(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="test_docs")
    store.add_chunks([embedded_chunk("chunk-0", "PTO is 15 days.", [1.0, 0.0])])

    store.clear()

    stats = store.stats()
    assert stats.collection == "test_docs"
    assert stats.count == 0
    assert stats.path == str(tmp_path / "chroma")
