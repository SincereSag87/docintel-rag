import json
from pathlib import Path

import pytest

from app.evaluation.dataset import EvaluationDatasetError, load_evaluation_dataset
from app.evaluation.models import RAGEvaluationDataset


def dataset_payload() -> dict:
    return {
        "name": "test",
        "documents": [{"filename": "doc.txt", "content": "PTO is 15 days."}],
        "cases": [
            {
                "id": "pto",
                "question": "How much PTO?",
                "answerable": True,
                "expected_answer_contains": ["15 days"],
                "expected_documents": ["doc.txt"],
            }
        ],
    }


def test_loads_valid_dataset(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(dataset_payload()), encoding="utf-8")

    dataset = load_evaluation_dataset(path)

    assert dataset.name == "test"
    assert dataset.cases[0].id == "pto"


def test_invalid_answerable_case_requires_expected_phrases() -> None:
    payload = dataset_payload()
    payload["cases"][0]["expected_answer_contains"] = []

    with pytest.raises(ValueError):
        RAGEvaluationDataset.model_validate(payload)


def test_missing_expected_document_is_invalid() -> None:
    payload = dataset_payload()
    payload["cases"][0]["expected_documents"] = ["missing.txt"]

    with pytest.raises(ValueError):
        RAGEvaluationDataset.model_validate(payload)


def test_duplicate_case_ids_are_invalid() -> None:
    payload = dataset_payload()
    payload["cases"].append(payload["cases"][0])

    with pytest.raises(ValueError):
        RAGEvaluationDataset.model_validate(payload)


def test_dataset_loader_wraps_json_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")

    with pytest.raises(EvaluationDatasetError):
        load_evaluation_dataset(path)
