import pytest

from app.rag.parsers import RAGParseError, parse_rag_response
from app.rag.prompts import INSUFFICIENT_INFORMATION_ANSWER


def test_parser_accepts_valid_json() -> None:
    parsed = parse_rag_response(
        '{"answer": "PTO is 15 days.", "answered": true, "source_numbers": [1]}',
        available_source_count=1,
    )

    assert parsed.answered is True
    assert parsed.source_numbers == [1]


def test_parser_accepts_fenced_json() -> None:
    parsed = parse_rag_response(
        '```json\n{"answer": "No.", "answered": false, "source_numbers": []}\n```',
        available_source_count=1,
    )

    assert parsed.answered is False


def test_parser_rejects_malformed_json() -> None:
    with pytest.raises(RAGParseError):
        parse_rag_response("not json", available_source_count=1)


def test_parser_rejects_missing_fields() -> None:
    with pytest.raises(RAGParseError):
        parse_rag_response('{"answer": "Missing fields"}', available_source_count=1)


def test_parser_rejects_invalid_source_number() -> None:
    with pytest.raises(RAGParseError):
        parse_rag_response(
            '{"answer": "PTO is 15 days.", "answered": true, "source_numbers": [2]}',
            available_source_count=1,
        )


def test_parser_rejects_citations_for_insufficient_information() -> None:
    with pytest.raises(RAGParseError):
        parse_rag_response(
            (
                '{"answer": "'
                + INSUFFICIENT_INFORMATION_ANSWER
                + '", "answered": false, "source_numbers": [1]}'
            ),
            available_source_count=1,
        )
