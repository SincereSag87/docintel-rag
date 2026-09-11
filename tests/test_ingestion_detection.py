from pathlib import Path

import pytest

from app.ingestion.base import DocumentNotFoundError, UnsupportedDocumentTypeError
from app.ingestion.document_ingestor import DocumentIngestor
from app.ingestion.models import DocumentFileType, detect_document_type


def test_detects_supported_file_types() -> None:
    assert detect_document_type(Path("handbook.pdf")) == DocumentFileType.PDF
    assert detect_document_type(Path("policy.DOCX")) == DocumentFileType.DOCX
    assert detect_document_type(Path("notes.txt")) == DocumentFileType.TXT


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedDocumentTypeError):
        detect_document_type(Path("archive.csv"))


def test_missing_file_validation(tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError):
        DocumentIngestor().ingest(tmp_path / "missing.pdf")


def test_directory_validation(tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError):
        DocumentIngestor().ingest(tmp_path)
