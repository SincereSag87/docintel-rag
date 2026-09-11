from collections.abc import Sequence
from http import HTTPStatus

from openai import APIConnectionError, APIError, NotFoundError, OpenAI

from app.core.config import Settings, get_settings
from app.llm.base import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMResponseError,
)
from app.llm.models import ChatMessage, ChatResponse


class OllamaProvider(LLMProvider):
    """LLM provider backed by Ollama's OpenAI-compatible chat endpoint."""

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or OpenAI(
            base_url=self.settings.ollama_base_url,
            api_key="ollama",
        )

    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str | None = None,
    ) -> ChatResponse:
        if not messages:
            raise ValueError("At least one chat message is required.")

        selected_model = model or self.settings.default_model
        payload = [message.model_dump() for message in messages]

        try:
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=payload,
            )
        except NotFoundError as exc:
            raise LLMModelNotFoundError(
                f"Ollama model '{selected_model}' was not found. Pull it with: "
                f"ollama pull {selected_model}"
            ) from exc
        except APIConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to Ollama at {self.settings.ollama_base_url}."
            ) from exc
        except APIError as exc:
            if getattr(exc, "status_code", None) == HTTPStatus.NOT_FOUND:
                raise LLMModelNotFoundError(
                    f"Ollama model '{selected_model}' was not found. Pull it with: "
                    f"ollama pull {selected_model}"
                ) from exc
            raise LLMResponseError("Ollama returned an error while generating a response.") from exc

        content = self._extract_content(response)
        return ChatResponse(model=selected_model, content=content)

    @staticmethod
    def _extract_content(response: object) -> str:
        try:
            choices = response.choices
            content = choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError("Ollama returned a malformed chat response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("Ollama returned an empty chat response.")

        return content.strip()
