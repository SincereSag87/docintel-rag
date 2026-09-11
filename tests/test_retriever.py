from app.documents.models import DocumentChunk, EmbeddedChunk
from app.embeddings.base import EmbeddingProvider
from app.retrieval.retriever import Retriever
from app.storage.base import VectorSearchResult, VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.search_call = None

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        pass

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        self.search_call = (query_embedding, top_k, document_ids)
        chunk = DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            text="PTO is 15 days.",
            chunk_index=0,
            metadata={"filename": "handbook.txt", "document_id": "doc-1", "chunk_index": 0},
        )
        return [
            VectorSearchResult(
                chunk=EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2]),
                distance=0.12,
                metadata=chunk.metadata,
            )
        ]

    def delete_document(self, document_id: str) -> None:
        pass

    def clear(self) -> None:
        pass


def test_retriever_embeds_query_and_propagates_top_k() -> None:
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()

    results = Retriever(provider, store).retrieve("pto days", top_k=3)

    assert provider.queries == ["pto days"]
    assert store.search_call == ([0.1, 0.2], 3, None)
    assert results[0].distance == 0.12
    assert results[0].filename == "handbook.txt"


def test_retriever_propagates_document_filter() -> None:
    store = FakeVectorStore()

    Retriever(FakeEmbeddingProvider(), store).retrieve("pto", document_ids=["doc-1"])

    assert store.search_call == ([0.1, 0.2], 5, ["doc-1"])
