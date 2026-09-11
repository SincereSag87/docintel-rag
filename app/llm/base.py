from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.llm.models import ChatMessage, ChatResponse


class LLMError(Exception):
    """Base exception for LLM provider failures."""


class LLMConnectionError(LLMError):
    """Raised when the LLM service cannot be reached."""


class LLMModelNotFoundError(LLMError):
    """Raised when the requested generation model is unavailable."""


class LLMResponseError(LLMError):
    """Raised when the LLM service returns an unusable response."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str | None = None,
    ) -> ChatResponse:
        """Generate a chat response from a sequence of messages."""
