from pathlib import Path

from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.models import RAGEvaluationReport
from app.evaluation.runner import RAGEvaluationRunner


class EvaluationService:
    def run(
        self,
        dataset_path: str | Path,
        model: str,
        retrieval_only: bool = False,
    ) -> RAGEvaluationReport:
        dataset = load_evaluation_dataset(dataset_path)
        return RAGEvaluationRunner(
            dataset=dataset,
            model=model,
            retrieval_only=retrieval_only,
        ).run()
