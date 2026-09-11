import argparse

from app.core.config import get_settings
from app.embeddings.base import EmbeddingError
from app.embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from app.llm.base import LLMError
from app.llm.models import ChatMessage
from app.llm.ollama_provider import OllamaProvider
from app.services.health_service import HealthService

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


def main() -> int:
    args = build_parser().parse_args()

    if args.llm_test:
        return run_llm_test(args.model)
    if args.embedding_test:
        return run_embedding_test()
    return print_health()


if __name__ == "__main__":
    raise SystemExit(main())
