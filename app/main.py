import argparse
import json

from app.core.config import get_settings
from app.embeddings.base import EmbeddingError
from app.embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from app.ingestion.base import DocumentIngestionError
from app.llm.base import LLMError
from app.llm.models import ChatMessage
from app.llm.ollama_provider import OllamaProvider
from app.services.document_service import DocumentService
from app.services.health_service import HealthService
from app.services.index_service import IndexService
from app.services.rag_service import RAGService

RAG_EXPLANATION_PROMPT = "Explain retrieval-augmented generation in three sentences."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocIntel RAG Phase 1 CLI")
    parser.add_argument("--llm-test", action="store_true", help="Run a live LLM smoke test.")
    parser.add_argument(
        "--embedding-test",
        action="store_true",
        help="Run an embedding smoke test.",
    )
    parser.add_argument("--model", help="Override the Ollama generation model for --llm-test.")
    parser.add_argument("--ingest", help="Ingest and summarize a local PDF, DOCX, or TXT file.")
    parser.add_argument(
        "--inspect",
        help="Inspect a local PDF, DOCX, or TXT file without persistence.",
    )
    parser.add_argument("--index", help="Index a local PDF, DOCX, or TXT file into ChromaDB.")
    parser.add_argument("--search", help="Run semantic vector search against the local index.")
    parser.add_argument("--top-k", type=int, help="Number of search results to return.")
    parser.add_argument(
        "--index-stats",
        action="store_true",
        help="Print local vector index stats.",
    )
    parser.add_argument("--delete-document", help="Delete one document from the vector index.")
    parser.add_argument("--clear-index", action="store_true", help="Clear the local vector index.")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive commands.")
    parser.add_argument("--ask", help="Ask a grounded question using retrieved indexed context.")
    parser.add_argument(
        "--document-id",
        action="append",
        help="Restrict --ask or --search to a document ID. Can be passed more than once.",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format for --ask.",
    )
    parser.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Show retrieved chunks and distances for --ask debugging.",
    )
    return parser


def print_health() -> int:
    settings = get_settings()
    provider = OllamaProvider(settings=settings)
    report = HealthService(settings=settings, llm_provider=provider).check()
    status = "reachable" if report.ollama_reachable else "unavailable"

    print("DocIntel RAG")
    print(f"Ollama: {status}")
    print(f"Generation model: {report.generation_model}")
    print(f"Embedding model: {report.embedding_model}")
    if report.error:
        print(f"Health detail: {report.error}")

    return 0 if report.ollama_reachable else 1


def run_llm_test(model: str | None = None) -> int:
    settings = get_settings()
    provider = OllamaProvider(settings=settings)
    selected_model = model or settings.default_model

    print("DocIntel RAG LLM Smoke Test")
    print(f"Generation model: {selected_model}")
    try:
        response = provider.generate(
            [ChatMessage(role="user", content=RAG_EXPLANATION_PROMPT)],
            model=selected_model,
        )
    except LLMError as exc:
        print(f"LLM test failed: {exc}")
        return 1

    print(response.content)
    return 0


def run_embedding_test() -> int:
    provider = SentenceTransformerEmbeddingProvider()

    print("DocIntel RAG Embedding Smoke Test")
    print(f"Embedding model: {provider.model_name}")
    try:
        vector = provider.embed_text("DocIntel turns documents into grounded answers.")
    except EmbeddingError as exc:
        print(f"Embedding test failed: {exc}")
        return 1

    print(f"Vector dimension: {len(vector)}")
    return 0


def run_document_ingestion(path: str, title: str = "DocIntel RAG Document Ingestion") -> int:
    service = DocumentService()
    try:
        summary = service.inspect_document(path)
    except DocumentIngestionError as exc:
        print(f"Document ingestion failed: {exc}")
        return 1

    print(title)
    print(f"Filename: {summary.filename}")
    print(f"Document ID: {summary.document_id}")
    print(f"Source type: {summary.source_type}")
    print(f"Characters: {summary.characters}")
    print(f"Words: {summary.words}")
    if "page_count" in summary.metadata:
        print(f"Pages: {summary.metadata['page_count']}")
    print("Metadata:")
    for key, value in sorted(summary.metadata.items()):
        print(f"  {key}: {value}")
    print("Preview:")
    print(summary.preview)
    return 0


def run_index(path: str) -> int:
    service = IndexService()
    try:
        result = service.index_document(path)
    except Exception as exc:
        print(f"Indexing failed: {exc}")
        return 1

    print("DocIntel RAG Indexing")
    print(f"Filename: {result.filename}")
    print(f"Document ID: {result.document_id}")
    print(f"Chunks indexed: {result.chunk_count}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Collection: {result.collection}")
    print(f"Elapsed seconds: {result.elapsed_seconds:.2f}")
    return 0


def run_search(
    query: str,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> int:
    service = IndexService()
    try:
        results = service.search(query, top_k=top_k, document_ids=document_ids)
    except Exception as exc:
        print(f"Search failed: {exc}")
        return 1

    print("DocIntel RAG Semantic Search")
    print(f"Query: {query}")
    if not results:
        print("No results found.")
        return 0

    for rank, result in enumerate(results, start=1):
        excerpt = result.text[:500].strip()
        if len(result.text) > 500:
            excerpt = f"{excerpt}..."
        print("")
        print(f"Rank {rank}")
        print(f"Filename: {result.filename}")
        print(f"Chunk: {result.chunk_index}")
        print(f"Distance: {result.distance:.6f}")
        print("Excerpt:")
        print(excerpt)
    return 0


def run_ask(
    question: str,
    model: str | None = None,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
    output: str = "text",
    show_retrieval: bool = False,
) -> int:
    service = RAGService()
    try:
        answer = service.ask(
            question=question,
            model=model,
            top_k=top_k,
            document_ids=document_ids,
        )
    except Exception as exc:
        print(f"RAG question answering failed: {exc}")
        return 1

    if output == "json":
        print(json.dumps(answer.model_dump(), indent=2))
        return 0

    print("Question:")
    print(answer.question)
    print("")
    print("Answer:")
    print(answer.answer)
    print("")
    print("Model:")
    print(answer.model)
    print("")
    print("Sources:")
    if not answer.citations:
        print("None")
    for index, citation in enumerate(answer.citations, start=1):
        print(f"[{index}] {citation.filename}")
        print(f"Chunk: {citation.chunk_index}")
        if citation.page is not None:
            print(f"Page: {citation.page}")
        print("Excerpt:")
        print(citation.excerpt)

    if show_retrieval:
        print("")
        print("Retrieval Evidence:")
        if not answer.retrieval_results:
            print("None")
        for index, result in enumerate(answer.retrieval_results, start=1):
            excerpt = result.text[:500].strip()
            if len(result.text) > 500:
                excerpt = f"{excerpt}..."
            print("")
            print(f"Rank {index}")
            print(f"Filename: {result.filename}")
            print(f"Chunk: {result.chunk_index}")
            print(f"Distance: {result.distance:.6f}")
            print("Excerpt:")
            print(excerpt)
    return 0


def run_index_stats() -> int:
    stats = IndexService().get_index_stats()
    print("DocIntel RAG Index Stats")
    print(f"Collection: {stats.collection}")
    print(f"Count: {stats.count}")
    print(f"Path: {stats.path}")
    return 0


def run_delete_document(document_id: str) -> int:
    IndexService().delete_document(document_id)
    print("DocIntel RAG Delete Document")
    print(f"Deleted document: {document_id}")
    return 0


def run_clear_index(confirmed: bool) -> int:
    if not confirmed:
        print("Clear index requires explicit confirmation: --clear-index --yes")
        return 1

    IndexService().clear_index()
    print("DocIntel RAG Clear Index")
    print("Index cleared.")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.llm_test:
        return run_llm_test(args.model)
    if args.embedding_test:
        return run_embedding_test()
    if args.ingest:
        return run_document_ingestion(args.ingest)
    if args.inspect:
        return run_document_ingestion(args.inspect, title="DocIntel RAG Document Inspection")
    if args.index:
        return run_index(args.index)
    if args.search:
        return run_search(args.search, top_k=args.top_k, document_ids=args.document_id)
    if args.ask:
        return run_ask(
            args.ask,
            model=args.model,
            top_k=args.top_k,
            document_ids=args.document_id,
            output=args.output,
            show_retrieval=args.show_retrieval,
        )
    if args.index_stats:
        return run_index_stats()
    if args.delete_document:
        return run_delete_document(args.delete_document)
    if args.clear_index:
        return run_clear_index(args.yes)
    return print_health()


if __name__ == "__main__":
    raise SystemExit(main())
