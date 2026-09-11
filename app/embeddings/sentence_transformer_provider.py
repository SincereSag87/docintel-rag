from typing import Any

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingModelError, EmbeddingProvider, EmbeddingResponseError


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers with lazy model loading."""

    def __init__(self, settings: Settings | None = None, model: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = model

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text to embed must not be empty.")

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("At least one text value is required.")
        if any(not text or not text.strip() for text in texts):
            raise ValueError("Text values to embed must not be empty.")

        model = self._get_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=False)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Embedding model '{self.model_name}' failed to generate vectors."
            ) from exc

        vectors = self._normalize_vectors(embeddings)
        if len(vectors) != len(texts):
            raise EmbeddingResponseError("Embedding provider returned the wrong number of vectors.")

        return vectors

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Could not load embedding model '{self.model_name}'."
                ) from exc
        return self._model

    @staticmethod
    def _normalize_vectors(embeddings: Any) -> list[list[float]]:
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        vectors: list[list[float]] = []
        try:
            for vector in embeddings:
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                normalized = [float(value) for value in vector]
                if not normalized:
                    raise ValueError
                vectors.append(normalized)
        except (TypeError, ValueError) as exc:
            raise EmbeddingResponseError("Embedding provider returned malformed vectors.") from exc

        if not vectors:
            raise EmbeddingResponseError("Embedding provider returned no vectors.")
        return vectors
