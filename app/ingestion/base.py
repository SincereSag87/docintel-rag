from abc import ABC, abstractmethod
from pathlib import Path

from app.documents.models import Document


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when a file extension is not supported."""


class DocumentNotFoundError(DocumentIngestionError):
    """Raised when an input path does not point to a readable file."""


class DocumentExtractionError(DocumentIngestionError):
    """Raised when a loader cannot extract usable text."""


class EmptyDocumentError(DocumentExtractionError):
    """Raised when a document has no meaningful text content."""


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> Document:
        """Load a document from a local file path."""
