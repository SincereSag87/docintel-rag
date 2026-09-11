from app.documents.models import Citation
from app.llm.base import LLMProvider
from app.rag.context import ContextBuilder
from app.rag.models import GroundedAnswer, RetrievedContext
from app.rag.parsers import parse_rag_response
from app.rag.prompts import INSUFFICIENT_INFORMATION_ANSWER, build_grounded_qa_messages
from app.retrieval.retriever import Retriever


class RAGEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.context_builder = context_builder or ContextBuilder()

    def answer(
        self,
        question: str,
        *,
        model: str | None = None,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> GroundedAnswer:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        results = self.retriever.retrieve(question, top_k=top_k, document_ids=document_ids)
        if not results:
            return GroundedAnswer(
                question=question,
                answer=INSUFFICIENT_INFORMATION_ANSWER,
                answered=False,
                citations=[],
                model=model or self.retriever.settings.default_model,
                retrieval_count=0,
                confidence_label="insufficient_evidence",
                retrieval_results=[],
            )

        context = self.context_builder.build(question, results)
        messages = build_grounded_qa_messages(question, context.formatted_context)
        response = self.llm_provider.generate(messages, model=model)
        parsed = parse_rag_response(
            response.content,
            available_source_count=context.used_result_count,
        )
        citations = self._citations_from_source_numbers(parsed.source_numbers, context)

        return GroundedAnswer(
            question=question,
            answer=parsed.answer,
            answered=parsed.answered,
            citations=citations,
            model=response.model,
            retrieval_count=context.used_result_count,
            confidence_label="grounded" if parsed.answered else "insufficient_evidence",
            retrieval_results=context.results,
        )

    @staticmethod
    def _citations_from_source_numbers(
        source_numbers: list[int],
        context: RetrievedContext,
    ) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[int] = set()
        for source_number in source_numbers:
            if source_number in seen:
                continue
            seen.add(source_number)
            result = context.results[source_number - 1]
            page = result.metadata.get("page")
            citations.append(
                Citation(
                    document_id=result.document_id,
                    filename=result.filename,
                    chunk_id=result.chunk_id,
                    chunk_index=result.chunk_index,
                    excerpt=result.text[:500].strip(),
                    page=int(page) if isinstance(page, int) else None,
                )
            )
        return citations
