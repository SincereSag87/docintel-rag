from app.core.config import Settings
from app.llm.base import LLMConnectionError, LLMProvider
from app.llm.models import ChatMessage, ChatResponse
from app.services.health_service import HealthService


class HealthyLLM(LLMProvider):
    def generate(self, messages: list[ChatMessage], model: str | None = None) -> ChatResponse:
        return ChatResponse(model=model or "llama3.2", content="OK")


class UnavailableLLM(LLMProvider):
    def generate(self, messages: list[ChatMessage], model: str | None = None) -> ChatResponse:
        raise LLMConnectionError("offline")


def test_health_reports_ollama_reachable() -> None:
    report = HealthService(settings=Settings(), llm_provider=HealthyLLM()).check()

    assert report.ollama_reachable is True
    assert report.generation_model == "llama3.2"
    assert report.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_health_reports_ollama_unavailable() -> None:
    report = HealthService(settings=Settings(), llm_provider=UnavailableLLM()).check()

    assert report.ollama_reachable is False
    assert report.error == "offline"


def test_health_reports_configuration_without_external_checks() -> None:
    settings = Settings(DEFAULT_MODEL="gemma3", EMBEDDING_MODEL="custom-embedding")

    report = HealthService(settings=settings).check()

    assert report.ollama_reachable is False
    assert report.generation_model == "gemma3"
    assert report.embedding_model == "custom-embedding"
    assert report.embedding_provider_loaded is None
