from app.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.default_model == "llama3.2"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 5


def test_settings_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11435/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "gemma3")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    monkeypatch.setenv("TOP_K", "8")

    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11435/v1"
    assert settings.default_model == "gemma3"
    assert settings.embedding_model == "custom-embedding"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.top_k == 8
