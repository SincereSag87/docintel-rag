from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, NotFoundError
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.base import LLMConnectionError, LLMModelNotFoundError, LLMResponseError
from app.llm.models import ChatMessage
from app.llm.ollama_provider import OllamaProvider


class FakeCompletions:
    def __init__(self, response=None, exception=None) -> None:
        self.response = response
        self.exception = exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exception:
            raise self.exception
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def response_with_content(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def provider_for(completions: FakeCompletions) -> OllamaProvider:
    return OllamaProvider(settings=Settings(), client=FakeClient(completions))


def test_generate_success() -> None:
    completions = FakeCompletions(
        response=response_with_content("RAG combines search and generation.")
    )
    provider = provider_for(completions)

    response = provider.generate([ChatMessage(role="user", content="What is RAG?")])

    assert response.model == "llama3.2"
    assert response.content == "RAG combines search and generation."
    assert completions.calls[0]["model"] == "llama3.2"


def test_generate_model_override() -> None:
    completions = FakeCompletions(response=response_with_content("Hello from Gemma."))
    provider = provider_for(completions)

    response = provider.generate([ChatMessage(role="user", content="Hi")], model="gemma3")

    assert response.model == "gemma3"
    assert completions.calls[0]["model"] == "gemma3"


def test_generate_unavailable_service() -> None:
    completions = FakeCompletions(exception=APIConnectionError(request=None))
    provider = provider_for(completions)

    with pytest.raises(LLMConnectionError):
        provider.generate([ChatMessage(role="user", content="Hi")])


def test_generate_missing_model() -> None:
    request = httpx2.Request("POST", "http://localhost:11434/v1/chat/completions")
    response = httpx2.Response(404, request=request)
    completions = FakeCompletions(
        exception=NotFoundError("missing", response=response, body=None)
    )
    provider = provider_for(completions)

    with pytest.raises(LLMModelNotFoundError):
        provider.generate([ChatMessage(role="user", content="Hi")], model="unknown")


def test_generate_empty_response() -> None:
    completions = FakeCompletions(response=response_with_content(""))
    provider = provider_for(completions)

    with pytest.raises(LLMResponseError):
        provider.generate([ChatMessage(role="user", content="Hi")])


def test_chat_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")
