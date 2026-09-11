import pytest

from app.documents.models import DocumentChunk
from app.embeddings.base import EmbeddingProvider, EmbeddingResponseError
from app.embeddings.pipeline import EmbeddingPipeline


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.batch_calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        return self.vectors[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(texts)
        return self.vectors


def chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(id="c1", document_id="d1", text="alpha", chunk_index=0),
        DocumentChunk(id="c2", document_id="d1", text="beta", chunk_index=1),
    ]


def test_embedding_pipeline_uses_batch_embedding() -> None:
    provider = FakeEmbeddingProvider([[0.1, 0.2], [0.3, 0.4]])
    embedded = EmbeddingPipeline(provider).embed_chunks(chunks())

    assert provider.batch_calls == [["alpha", "beta"]]
    assert [item.embedding for item in embedded] == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_pipeline_handles_empty_chunks() -> None:
    provider = FakeEmbeddingProvider([])

    assert EmbeddingPipeline(provider).embed_chunks([]) == []


def test_embedding_pipeline_rejects_vector_count_mismatch() -> None:
    provider = FakeEmbeddingProvider([[0.1, 0.2]])

    with pytest.raises(EmbeddingResponseError):
        EmbeddingPipeline(provider).embed_chunks(chunks())


def test_embedding_pipeline_rejects_inconsistent_dimensions() -> None:
    provider = FakeEmbeddingProvider([[0.1, 0.2], [0.3]])

    with pytest.raises(EmbeddingResponseError):
        EmbeddingPipeline(provider).embed_chunks(chunks())
