from app.chunking.base import Chunker
from app.core.config import Settings
from app.documents.models import Document, DocumentChunk, EmbeddedChunk
from app.embeddings.base import EmbeddingProvider
from app.services.index_service import IndexService
from app.storage.base import VectorSearchResult, VectorStore
from app.storage.models import VectorStoreStats


class FakeChunker(Chunker):
    def chunk(self, document: Document) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                id=f"{document.id}:chunk:0:test",
                document_id=document.id,
                text=document.text,
                chunk_index=0,
                metadata={
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_index": 0,
                },
            )
        ]


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.text_batches: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_batches.append(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.chunks: dict[str, EmbeddedChunk] = {}
        self.deleted: list[str] = []
        self.cleared = False

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.chunk.id] = chunk

    def search(self, query_embedding: list[float], top_k: int) -> list[VectorSearchResult]:
        return []

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)

    def clear(self) -> None:
        self.cleared = True
        self.chunks.clear()

    def stats(self) -> VectorStoreStats:
        return VectorStoreStats(collection="test", count=len(self.chunks), path=None)


def make_service(store: FakeVectorStore, provider: FakeEmbeddingProvider) -> IndexService:
    return IndexService(
        chunker=FakeChunker(),
        embedding_provider=provider,
        vector_store=store,
        settings=Settings(CHROMA_COLLECTION="test"),
    )


def test_index_service_indexes_document_model() -> None:
    store = FakeVectorStore()
    provider = FakeEmbeddingProvider()
    service = make_service(store, provider)
    document = Document(id="doc-1", filename="handbook.txt", text="PTO is 15 days.")

    result = service.index_document_model(document)

    assert result.document_id == "doc-1"
    assert result.filename == "handbook.txt"
    assert result.chunk_count == 1
    assert result.collection == "test"
    assert provider.text_batches == [["PTO is 15 days."]]
    assert len(store.chunks) == 1


def test_index_service_repeated_indexing_is_idempotent_with_upsert_store() -> None:
    store = FakeVectorStore()
    provider = FakeEmbeddingProvider()
    service = make_service(store, provider)
    document = Document(id="doc-1", filename="handbook.txt", text="PTO is 15 days.")

    service.index_document_model(document)
    service.index_document_model(document)

    assert len(store.chunks) == 1


def test_index_service_delete_clear_and_stats() -> None:
    store = FakeVectorStore()
    service = make_service(store, FakeEmbeddingProvider())

    service.delete_document("doc-1")
    service.clear_index()
    stats = service.get_index_stats()

    assert store.deleted == ["doc-1"]
    assert store.cleared is True
    assert stats.count == 0
