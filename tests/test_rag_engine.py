from collections.abc import Sequence

import pytest

from app.core.config import Settings
from app.llm.base import LLMConnectionError, LLMProvider
from app.llm.models import ChatMessage, ChatResponse
from app.rag.context import ContextBuilder
from app.rag.engine import RAGEngine
from app.rag.parsers import RAGParseError
from app.rag.prompts import INSUFFICIENT_INFORMATION_ANSWER
from app.retrieval.models import RetrievalResult


def retrieval_result(index: int = 0, filename: str = "handbook.txt") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        filename=filename,
        text="Employees receive 15 days of paid time off.",
        distance=0.2,
        chunk_index=index,
        metadata={
            "filename": filename,
            "document_id": f"doc-{index}",
            "chunk_index": index,
            "page": 2,
        },
    )


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = []
        self.settings = Settings()

    def retrieve(self, query: str, top_k=None, document_ids=None):
        self.calls.append((query, top_k, document_ids))
        return self.results


class FakeLLMProvider(LLMProvider):
    def __init__(self, content: str, model: str = "llama3.2") -> None:
        self.content = content
        self.model = model
        self.calls = []

    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str | None = None,
    ) -> ChatResponse:
        self.calls.append((messages, model))
        return ChatResponse(model=model or self.model, content=self.content)


class FailingLLMProvider(LLMProvider):
    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str | None = None,
    ) -> ChatResponse:
        raise LLMConnectionError("offline")


def test_rag_engine_returns_grounded_answer_and_real_citations() -> None:
    retriever = FakeRetriever([retrieval_result()])
    llm = FakeLLMProvider(
        '{"answer": "Employees receive 15 days of paid time off.", '
        '"answered": true, "source_numbers": [1]}'
    )

    answer = RAGEngine(retriever, llm, ContextBuilder(max_chars=5000)).answer("How many PTO days?")

    assert answer.answered is True
    assert answer.citations[0].filename == "handbook.txt"
    assert answer.citations[0].document_id == "doc-0"
    assert answer.citations[0].chunk_id == "chunk-0"
    assert answer.citations[0].page == 2


def test_rag_engine_citation_safety_ignores_model_invented_metadata() -> None:
    result = retrieval_result(filename="trusted.txt")
    llm = FakeLLMProvider(
        '{"answer": "PTO is 15 days from invented.pdf.", '
        '"answered": true, "source_numbers": [1], "filename": "invented.pdf"}'
    )

    answer = RAGEngine(FakeRetriever([result]), llm, ContextBuilder(max_chars=5000)).answer("PTO?")

    assert answer.citations[0].filename == "trusted.txt"
    assert answer.citations[0].document_id == "doc-0"


def test_rag_engine_empty_retrieval_bypasses_llm() -> None:
    retriever = FakeRetriever([])
    llm = FakeLLMProvider('{"answer": "should not call", "answered": true, "source_numbers": []}')

    answer = RAGEngine(retriever, llm).answer("Unknown?")

    assert answer.answered is False
    assert answer.answer == INSUFFICIENT_INFORMATION_ANSWER
    assert answer.citations == []
    assert llm.calls == []


def test_rag_engine_insufficient_information_response() -> None:
    llm = FakeLLMProvider(
        '{"answer": "'
        + INSUFFICIENT_INFORMATION_ANSWER
        + '", "answered": false, "source_numbers": []}'
    )

    answer = RAGEngine(FakeRetriever([retrieval_result()]), llm).answer("Insurance?")

    assert answer.answered is False
    assert answer.citations == []


def test_rag_engine_propagates_model_top_k_and_document_filter() -> None:
    retriever = FakeRetriever([retrieval_result()])
    llm = FakeLLMProvider(
        '{"answer": "Employees receive 15 days.", "answered": true, "source_numbers": [1]}'
    )

    answer = RAGEngine(retriever, llm).answer(
        "PTO?",
        model="gemma3",
        top_k=3,
        document_ids=["doc-0"],
    )

    assert answer.model == "gemma3"
    assert retriever.calls == [("PTO?", 3, ["doc-0"])]
    assert llm.calls[0][1] == "gemma3"


def test_rag_engine_raises_parse_failure() -> None:
    llm = FakeLLMProvider("not json")

    with pytest.raises(RAGParseError):
        RAGEngine(FakeRetriever([retrieval_result()]), llm).answer("PTO?")


def test_rag_engine_propagates_provider_failure() -> None:
    with pytest.raises(LLMConnectionError):
        RAGEngine(FakeRetriever([retrieval_result()]), FailingLLMProvider()).answer("PTO?")
