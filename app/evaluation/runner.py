import shutil
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

from app.core.config import Settings
from app.evaluation.answer_metrics import evaluate_answer
from app.evaluation.models import (
    EvaluationCase,
    RAGEvaluationCaseResult,
    RAGEvaluationDataset,
    RAGEvaluationReport,
)
from app.evaluation.retrieval_metrics import evaluate_retrieval
from app.rag.parsers import RAGParseError
from app.services.index_service import IndexService
from app.services.rag_service import RAGService


class RAGEvaluationRunner:
    def __init__(
        self,
        dataset: RAGEvaluationDataset,
        model: str,
        retrieval_only: bool = False,
        settings: Settings | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.retrieval_only = retrieval_only
        self.base_settings = settings or Settings()

    def run(self) -> RAGEvaluationReport:
        tmpdir = mkdtemp(prefix="docintel_eval_")
        index_service: IndexService | None = None
        try:
            root = Path(tmpdir)
            document_dir = root / "documents"
            chroma_path = root / "chroma"
            document_dir.mkdir()
            settings = self.base_settings.model_copy(update={"chroma_path": str(chroma_path)})
            self._write_documents(document_dir)

            index_service = IndexService(settings=settings)
            for document in self.dataset.documents:
                index_service.index_document(document_dir / document.filename)

            rag_service = RAGService(
                embedding_provider=index_service.embedding_provider,
                vector_store=index_service.vector_store,
                settings=settings,
            )
            case_results = [self._run_case(case, rag_service) for case in self.dataset.cases]
            return self._aggregate(case_results)
        finally:
            if index_service is not None:
                close = getattr(index_service.vector_store, "close", None)
                if callable(close):
                    close()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _write_documents(self, document_dir: Path) -> None:
        for document in self.dataset.documents:
            (document_dir / document.filename).write_text(document.content, encoding="utf-8")

    def _run_case(self, case: EvaluationCase, rag_service: RAGService) -> RAGEvaluationCaseResult:
        started = perf_counter()
        retrieval_latency_ms = 0.0
        generation_latency_ms = 0.0
        try:
            retrieval_started = perf_counter()
            retrieval_results = rag_service.retrieve(case.question, top_k=case.top_k)
            retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000
            retrieval_metrics = evaluate_retrieval(retrieval_results, case.expected_documents)

            if self.retrieval_only:
                success = self._retrieval_only_success(case, retrieval_metrics.hit)
                return self._case_result(
                    case=case,
                    success=success,
                    retrieval_metrics=retrieval_metrics,
                    latency_ms=(perf_counter() - started) * 1000,
                    retrieval_latency_ms=retrieval_latency_ms,
                    parse_valid=True,
                )

            generation_started = perf_counter()
            grounded_answer = rag_service.ask(case.question, model=self.model, top_k=case.top_k)
            generation_latency_ms = (perf_counter() - generation_started) * 1000
            answer_metrics = evaluate_answer(
                case,
                grounded_answer.answer,
                grounded_answer.answered,
                grounded_answer.citations,
            )
            success = self._case_success(case, retrieval_metrics.hit, answer_metrics)
            return self._case_result(
                case=case,
                success=success,
                retrieval_metrics=retrieval_metrics,
                answer=grounded_answer.answer,
                answered=grounded_answer.answered,
                cited_documents=[citation.filename for citation in grounded_answer.citations],
                answer_metrics=answer_metrics,
                latency_ms=(perf_counter() - started) * 1000,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                parse_valid=True,
            )
        except Exception as exc:
            parse_valid = not isinstance(exc, RAGParseError)
            return RAGEvaluationCaseResult(
                case_id=case.id,
                question=case.question,
                model=self.model,
                success=False,
                answerable=case.answerable,
                expected_documents=case.expected_documents,
                parse_valid=parse_valid,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                latency_ms=(perf_counter() - started) * 1000,
                error=str(exc),
            )

    def _case_result(
        self,
        *,
        case: EvaluationCase,
        success: bool,
        retrieval_metrics,
        answer: str = "",
        answered: bool | None = None,
        cited_documents: list[str] | None = None,
        answer_metrics=None,
        latency_ms: float,
        retrieval_latency_ms: float,
        generation_latency_ms: float = 0.0,
        parse_valid: bool,
    ) -> RAGEvaluationCaseResult:
        cited_documents = cited_documents or []
        return RAGEvaluationCaseResult(
            case_id=case.id,
            question=case.question,
            model=self.model,
            success=success,
            answerable=case.answerable,
            answered=answered,
            answer=answer,
            expected_documents=case.expected_documents,
            retrieved_documents=retrieval_metrics.retrieved_documents,
            cited_documents=cited_documents,
            retrieval_hit=retrieval_metrics.hit,
            retrieval_recall=retrieval_metrics.recall,
            reciprocal_rank=retrieval_metrics.reciprocal_rank,
            expected_document_coverage=retrieval_metrics.expected_document_coverage,
            answer_correct=bool(answer_metrics and answer_metrics.answer_correct),
            citation_precision=0.0 if answer_metrics is None else answer_metrics.citation_precision,
            citation_recall=0.0 if answer_metrics is None else answer_metrics.citation_recall,
            citation_correct=bool(answer_metrics and answer_metrics.citation_correct),
            unknown_behavior_correct=None
            if answer_metrics is None
            else answer_metrics.unknown_behavior_correct,
            parse_valid=parse_valid,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _case_success(case: EvaluationCase, retrieval_hit: bool, answer_metrics) -> bool:
        if case.answerable:
            return (
                retrieval_hit
                and answer_metrics.answer_correct
                and answer_metrics.citation_recall == 1.0
            )
        return bool(answer_metrics.unknown_behavior_correct)

    @staticmethod
    def _retrieval_only_success(case: EvaluationCase, retrieval_hit: bool) -> bool:
        return retrieval_hit if case.answerable else True

    def _aggregate(self, case_results: list[RAGEvaluationCaseResult]) -> RAGEvaluationReport:
        total = len(case_results)
        answerable = [result for result in case_results if result.answerable]
        unknown = [result for result in case_results if not result.answerable]
        return RAGEvaluationReport(
            dataset_name=self.dataset.name,
            model=self.model,
            retrieval_only=self.retrieval_only,
            cases_total=total,
            cases_passed=sum(result.success for result in case_results),
            retrieval_hit_rate=self._mean([result.retrieval_hit for result in answerable]),
            mean_retrieval_recall=self._mean([result.retrieval_recall for result in answerable]),
            mrr=self._mean([result.reciprocal_rank for result in answerable]),
            answer_accuracy=self._mean([result.answer_correct for result in answerable]),
            average_citation_precision=self._mean(
                [result.citation_precision for result in answerable]
            ),
            average_citation_recall=self._mean([result.citation_recall for result in answerable]),
            citation_accuracy=self._mean([result.citation_correct for result in answerable]),
            unknown_answer_accuracy=self._mean(
                [bool(result.unknown_behavior_correct) for result in unknown]
            ),
            parse_success_rate=self._mean([result.parse_valid for result in case_results]),
            average_latency_ms=self._mean([result.latency_ms for result in case_results]),
            failed_case_count=sum(not result.success for result in case_results),
            case_results=case_results,
        )

    @staticmethod
    def _mean(values) -> float:
        values = list(values)
        if not values:
            return 0.0
        return sum(float(value) for value in values) / len(values)
