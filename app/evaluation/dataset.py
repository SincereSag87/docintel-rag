import json
from pathlib import Path

from pydantic import ValidationError

from app.evaluation.models import RAGEvaluationDataset


class EvaluationDatasetError(Exception):
    """Raised when an evaluation dataset cannot be loaded or validated."""


def load_evaluation_dataset(path: str | Path) -> RAGEvaluationDataset:
    dataset_path = Path(path)
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationDatasetError(f"Could not read evaluation dataset: {dataset_path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset is not valid JSON: {dataset_path}"
        ) from exc

    try:
        return RAGEvaluationDataset.model_validate(data)
    except ValidationError as exc:
        raise EvaluationDatasetError(f"Evaluation dataset failed validation: {exc}") from exc
