from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.rag.engine import RAGEngine
from app.rag.models import GroundedAnswer
from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import Retriever
from app.storage.base import VectorStore
from app.storage.chroma_store import ChromaVectorStore


class RAGService:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_provider: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            settings=self.settings
        )
        self.vector_store = vector_store or ChromaVectorStore(settings=self.settings)
        self.retriever = retriever or Retriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            settings=self.settings,
        )
        self.llm_provider = llm_provider or OllamaProvider(settings=self.settings)
        self.engine = RAGEngine(retriever=self.retriever, llm_provider=self.llm_provider)

    def ask(
        self,
        question: str,
        model: str | None = None,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> GroundedAnswer:
        return self.engine.answer(
            question=question,
            model=model,
            top_k=top_k,
            document_ids=document_ids,
        )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        return self.retriever.retrieve(question, top_k=top_k, document_ids=document_ids)
