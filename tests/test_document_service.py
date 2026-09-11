from pathlib import Path

from app.services.document_service import DocumentService


def test_document_service_returns_inspection_summary(sample_txt_path: Path) -> None:
    summary = DocumentService().inspect_document(sample_txt_path)

    assert summary.filename == "sample.txt"
    assert summary.document_id.startswith("sha256:")
    assert summary.characters > 0
    assert summary.words > 0
    assert "Sample text" in summary.preview
    assert summary.metadata["document_type"] == "txt"
