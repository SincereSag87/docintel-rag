from app.core.config import Settings, get_settings
from app.rag.models import RetrievedContext
from app.retrieval.models import RetrievalResult


class ContextBuilder:
    def __init__(self, settings: Settings | None = None, max_chars: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.max_chars = max_chars or self.settings.max_rag_context_chars

    def build(self, query: str, results: list[RetrievalResult]) -> RetrievedContext:
        sections: list[str] = []
        used_results: list[RetrievalResult] = []
        total = 0

        for index, result in enumerate(results, start=1):
            section = self._format_source(index, result)
            projected = total + len(section) + (2 if sections else 0)
            if projected > self.max_chars and sections:
                break
            if projected > self.max_chars:
                section = section[: self.max_chars]
                projected = len(section)

            sections.append(section)
            used_results.append(result)
            total = projected

        formatted_context = "\n\n".join(sections)
        return RetrievedContext(
            query=query,
            results=used_results,
            formatted_context=formatted_context,
            total_characters=len(formatted_context),
            used_result_count=len(used_results),
        )

    @staticmethod
    def _format_source(source_number: int, result: RetrievalResult) -> str:
        page = result.metadata.get("page")
        page_line = f"Page: {page}\n" if page is not None else ""
        return (
            f"[SOURCE {source_number}]\n"
            f"Filename: {result.filename}\n"
            f"Document ID: {result.document_id}\n"
            f"Chunk ID: {result.chunk_id}\n"
            f"Chunk: {result.chunk_index}\n"
            f"{page_line}"
            f"Distance: {result.distance:.6f}\n"
            "Content:\n"
            f"{result.text}"
        )
