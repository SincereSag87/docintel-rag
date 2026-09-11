from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider
from app.llm.models import ChatMessage


class HealthReport(BaseModel):
    ollama_reachable: bool
    generation_model: str
    embedding_model: str
    embedding_provider_loaded: bool | None = None
    embedding_test_passed: bool | None = None
    error: str | None = None


class HealthService:
    def __init__(
        self,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

    def check(self, include_embedding_test: bool = False) -> HealthReport:
        error: str | None = None
        ollama_reachable = False

        if self.llm_provider is not None:
            try:
                self.llm_provider.generate(
                    [ChatMessage(role="user", content="Reply with OK.")],
                    model=self.settings.default_model,
                )
                ollama_reachable = True
            except Exception as exc:
                error = str(exc)

        embedding_provider_loaded = self._embedding_loaded()
        embedding_test_passed: bool | None = None

        if include_embedding_test and self.embedding_provider is not None:
            try:
                self.embedding_provider.embed_text("DocIntel health check")
                embedding_test_passed = True
                embedding_provider_loaded = True
            except Exception as exc:
                embedding_test_passed = False
                error = str(exc)

        return HealthReport(
            ollama_reachable=ollama_reachable,
            generation_model=self.settings.default_model,
            embedding_model=self.settings.embedding_model,
            embedding_provider_loaded=embedding_provider_loaded,
            embedding_test_passed=embedding_test_passed,
            error=error,
        )

    def _embedding_loaded(self) -> bool | None:
        if self.embedding_provider is None:
            return None
        return bool(getattr(self.embedding_provider, "is_loaded", False))
