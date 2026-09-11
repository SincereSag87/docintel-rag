from pathlib import Path

import pytest

from app.ingestion.base import EmptyDocumentError
from app.ingestion.docx_loader import DocxLoader


def test_docx_loader_extracts_paragraphs_and_headings(sample_docx_path: Path) -> None:
    document = DocxLoader().load(sample_docx_path)

    assert "PTO Policy" in document.text
    assert "Employees receive fifteen PTO days" in document.text
    assert document.metadata["paragraph_count"] >= 2
    assert document.metadata["title"] == "Employee Policy"


def test_docx_loader_extracts_simple_tables(sample_docx_path: Path) -> None:
    document = DocxLoader().load(sample_docx_path)

    assert "Expense | Approval" in document.text
    assert "Travel | Manager" in document.text
    assert document.metadata["table_count"] == 1


def test_docx_loader_rejects_empty_document(empty_docx_path: Path) -> None:
    with pytest.raises(EmptyDocumentError):
        DocxLoader().load(empty_docx_path)
