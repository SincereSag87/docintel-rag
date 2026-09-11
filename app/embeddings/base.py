from abc import ABC, abstractmethod


class EmbeddingError(Exception):
    """Base exception for embedding provider failures."""


class EmbeddingModelError(EmbeddingError):
    """Raised when an embedding model cannot load or run."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when an embedding provider returns invalid vectors."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
