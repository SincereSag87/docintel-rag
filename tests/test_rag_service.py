from app.rag.models import GroundedAnswer
from app.rag.prompts import INSUFFICIENT_INFORMATION_ANSWER
from app.retrieval.models import RetrievalResult
from app.services.rag_service import RAGService


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def answer(self, question, model=None, top_k=None, document_ids=None):
        self.calls.append((question, model, top_k, document_ids))
        return GroundedAnswer(
            question=question,
            answer=INSUFFICIENT_INFORMATION_ANSWER,
            answered=False,
            citations=[],
            model=model or "llama3.2",
            retrieval_count=0,
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, query, top_k=None, document_ids=None):
        self.calls.append((query, top_k, document_ids))
        return []


def test_rag_service_ask_delegates_to_engine() -> None:
    service = RAGService.__new__(RAGService)
    service.engine = FakeEngine()

    answer = service.ask("Insurance?", model="gemma3", top_k=2, document_ids=["doc-1"])

    assert answer.model == "gemma3"
    assert service.engine.calls == [("Insurance?", "gemma3", 2, ["doc-1"])]


def test_rag_service_retrieve_delegates_to_retriever() -> None:
    service = RAGService.__new__(RAGService)
    service.retriever = FakeRetriever()

    results: list[RetrievalResult] = service.retrieve("PTO?", top_k=1, document_ids=["doc-1"])

    assert results == []
    assert service.retriever.calls == [("PTO?", 1, ["doc-1"])]
