from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict

from app.chunking.base import Chunker
from app.chunking.recursive_chunker import RecursiveTextChunker
from app.core.config import Settings, get_settings
from app.documents.models import Document, EmbeddedChunk
from app.embeddings.base import EmbeddingProvider, EmbeddingResponseError
from app.embeddings.pipeline import EmbeddingPipeline
from app.embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from app.ingestion.document_ingestor import DocumentIngestor
from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import Retriever
from app.storage.base import VectorStore
from app.storage.chroma_store import ChromaVectorStore
from app.storage.models import VectorStoreStats


class IndexingResult(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    embedding_model: str
    collection: str
    elapsed_seconds: float

    model_config = ConfigDict(frozen=True)


class IndexService:
    def __init__(
        self,
        ingestor: DocumentIngestor | None = None,
        chunker: Chunker | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ingestor = ingestor or DocumentIngestor()
        self.chunker = chunker or RecursiveTextChunker(settings=self.settings)
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            settings=self.settings
        )
        self.embedding_pipeline = EmbeddingPipeline(self.embedding_provider)
        self.vector_store = vector_store or ChromaVectorStore(settings=self.settings)

    def index_document(self, path: str | Path) -> IndexingResult:
        return self.index_document_model(self.ingestor.ingest(path))

    def index_document_model(self, document: Document) -> IndexingResult:
        started = perf_counter()
        chunks = self.chunker.chunk(document)
        embedded_chunks = self.embedding_pipeline.embed_chunks(chunks)
        self._validate_embedded_chunks(embedded_chunks)
        self.vector_store.add_chunks(embedded_chunks)
        elapsed = perf_counter() - started

        stats = self.vector_store.stats()
        return IndexingResult(
            document_id=document.id,
            filename=document.filename,
            chunk_count=len(embedded_chunks),
            embedding_model=self.settings.embedding_model,
            collection=stats.collection,
            elapsed_seconds=elapsed,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        retriever = Retriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            settings=self.settings,
        )
        return retriever.retrieve(query=query, top_k=top_k, document_ids=document_ids)

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)

    def clear_index(self) -> None:
        self.vector_store.clear()

    def get_index_stats(self) -> VectorStoreStats:
        return self.vector_store.stats()

    @staticmethod
    def _validate_embedded_chunks(chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            raise EmbeddingResponseError("No embedded chunks were produced.")
