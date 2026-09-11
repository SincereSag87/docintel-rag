from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.documents.models import EmbeddedChunk


class VectorSearchResult(BaseModel):
    chunk: EmbeddedChunk
    score: float
    metadata: dict[str, object] = Field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the vector store."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[VectorSearchResult]:
        """Return the most relevant chunks for a query embedding."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks associated with a document."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed vectors."""
