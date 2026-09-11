from inspect import signature

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.retrieval.models import RetrievalResult
from app.storage.base import VectorStore


class Retriever:
    """Embeds a query and searches a vector store. Scores are Chroma distances."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")

        query_embedding = self.embedding_provider.embed_text(query)
        search_kwargs = {"query_embedding": query_embedding, "top_k": top_k or self.settings.top_k}
        if document_ids and self._store_accepts_document_filter():
            search_kwargs["document_ids"] = document_ids

        results = self.vector_store.search(**search_kwargs)
        return [
            RetrievalResult(
                chunk_id=result.chunk.chunk.id,
                document_id=result.chunk.chunk.document_id,
                filename=str(result.metadata.get("filename", "")),
                text=result.chunk.chunk.text,
                distance=result.distance,
                chunk_index=result.chunk.chunk.chunk_index,
                metadata=result.metadata,
            )
            for result in results
        ]

    def _store_accepts_document_filter(self) -> bool:
        return "document_ids" in signature(self.vector_store.search).parameters
