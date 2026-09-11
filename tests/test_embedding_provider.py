import sys

import pytest

from app.core.config import Settings
from app.embeddings.base import EmbeddingModelError, EmbeddingResponseError
from app.embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingProvider


class FakeModel:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def encode(self, texts, convert_to_numpy=False):
        self.calls.append((texts, convert_to_numpy))
        return self.result


class FailingModel:
    def encode(self, texts, convert_to_numpy=False):
        raise RuntimeError("boom")


def test_provider_uses_lazy_model_loading(monkeypatch) -> None:
    sentinel = object()

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            assert model_name == "sentence-transformers/all-MiniLM-L6-v2"

        def encode(self, texts, convert_to_numpy=False):
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        type("Module", (), {"SentenceTransformer": FakeSentenceTransformer}),
    )

    provider = SentenceTransformerEmbeddingProvider(settings=Settings())

    assert provider._model is None
    assert provider.embed_text("hello") == [0.1, 0.2, 0.3]
    assert provider._model is not sentinel


def test_embed_single_text() -> None:
    model = FakeModel([[1, 2, 3]])
    provider = SentenceTransformerEmbeddingProvider(model=model)

    assert provider.embed_text("hello") == [1.0, 2.0, 3.0]
    assert model.calls == [(["hello"], False)]


def test_embed_multiple_texts() -> None:
    model = FakeModel([[1, 2], [3, 4]])
    provider = SentenceTransformerEmbeddingProvider(model=model)

    assert provider.embed_texts(["hello", "world"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_empty_input_validation() -> None:
    provider = SentenceTransformerEmbeddingProvider(model=FakeModel([[1]]))

    with pytest.raises(ValueError):
        provider.embed_text("")
    with pytest.raises(ValueError):
        provider.embed_texts([])
    with pytest.raises(ValueError):
        provider.embed_texts(["valid", ""])


def test_model_failure() -> None:
    provider = SentenceTransformerEmbeddingProvider(model=FailingModel())

    with pytest.raises(EmbeddingModelError):
        provider.embed_text("hello")


def test_malformed_embedding_results() -> None:
    provider = SentenceTransformerEmbeddingProvider(model=FakeModel([["bad"]]))

    with pytest.raises(EmbeddingResponseError):
        provider.embed_text("hello")
