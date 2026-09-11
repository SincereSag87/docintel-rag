import json
import re

from pydantic import ValidationError

from app.rag.models import ParsedRAGResponse


class RAGParseError(Exception):
    """Raised when an LLM response cannot be parsed as a safe RAG answer."""


FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_rag_response(content: str, available_source_count: int) -> ParsedRAGResponse:
    raw = _strip_code_fence(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RAGParseError("RAG model response was not valid JSON.") from exc

    try:
        parsed = ParsedRAGResponse.model_validate(data)
    except ValidationError as exc:
        raise RAGParseError("RAG model response did not match the expected schema.") from exc

    invalid = [number for number in parsed.source_numbers if number > available_source_count]
    if invalid:
        raise RAGParseError(f"RAG model referenced unavailable source numbers: {invalid}.")
    if not parsed.answered and parsed.source_numbers:
        raise RAGParseError("Insufficient-information responses must not include citations.")
    return parsed


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    match = FENCED_JSON_PATTERN.fullmatch(stripped)
    if match:
        return match.group(1).strip()
    return stripped
