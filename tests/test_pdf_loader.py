from pathlib import Path

import pytest

from app.ingestion.base import DocumentExtractionError
from app.ingestion.pdf_loader import PDFLoader


def test_pdf_loader_extracts_page_text(sample_pdf_path: Path) -> None:
    document = PDFLoader().load(sample_pdf_path)

    assert "[Page 1]" in document.text
    assert "Employees must use MFA" in document.text


def test_pdf_loader_sets_page_count_metadata(sample_pdf_path: Path) -> None:
    document = PDFLoader().load(sample_pdf_path)

    assert document.metadata["page_count"] == 1
    assert document.metadata["file_type"] == "pdf"


def test_pdf_loader_handles_optional_metadata(sample_pdf_path: Path) -> None:
    document = PDFLoader().load(sample_pdf_path)

    assert document.metadata["author"] == "DocIntel"
    assert document.metadata["title"] == "Security Policy"
    assert document.metadata["subject"] == "Portfolio fixture"


def test_pdf_loader_rejects_empty_or_scanned_style_pdf(empty_pdf_path: Path) -> None:
    with pytest.raises(DocumentExtractionError, match="scanned or image-based"):
        PDFLoader().load(empty_pdf_path)
