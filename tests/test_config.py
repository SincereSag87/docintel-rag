import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.default_model == "llama3.2"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 5
    assert settings.chroma_path == "./data/chroma"
    assert settings.chroma_collection == "docintel"
    assert settings.max_rag_context_chars == 12000


def test_settings_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11435/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "gemma3")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("CHROMA_PATH", "./runtime/test-chroma")
    monkeypatch.setenv("CHROMA_COLLECTION", "test-docintel")
    monkeypatch.setenv("MAX_RAG_CONTEXT_CHARS", "6000")

    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11435/v1"
    assert settings.default_model == "gemma3"
    assert settings.embedding_model == "custom-embedding"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.top_k == 8
    assert settings.chroma_path == "./runtime/test-chroma"
    assert settings.chroma_collection == "test-docintel"
    assert settings.max_rag_context_chars == 6000


def test_settings_rejects_overlap_greater_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(CHUNK_SIZE=100, CHUNK_OVERLAP=100)
