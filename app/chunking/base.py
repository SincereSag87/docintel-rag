from abc import ABC, abstractmethod

from app.documents.models import Document, DocumentChunk


class ChunkingError(Exception):
    """Raised when document chunking fails."""


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split a document into stable, ordered chunks."""
