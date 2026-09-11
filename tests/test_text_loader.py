from pathlib import Path

import pytest

from app.ingestion.base import EmptyDocumentError
from app.ingestion.document_ingestor import DocumentIngestor
from app.ingestion.text_loader import TextLoader


def test_text_loader_extracts_text(sample_txt_path: Path) -> None:
    document = TextLoader().load(sample_txt_path)

    assert document.filename == "sample.txt"
    assert "Sample" in document.text
    assert document.metadata["encoding"] == "utf-8-sig"


def test_text_ingestion_normalizes_whitespace(sample_txt_path: Path) -> None:
    document = DocumentIngestor().ingest(sample_txt_path)

    assert "Sample text" in document.text
    assert "\n\nSecond paragraph" in document.text
    assert "\r" not in document.text


def test_text_loader_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        TextLoader().load(path)


def test_text_loader_handles_encoding_fallback(tmp_path: Path) -> None:
    path = tmp_path / "windows.txt"
    path.write_bytes(b"Manager approval costs \x96 required")

    document = TextLoader().load(path)

    assert "Manager approval" in document.text
    assert document.metadata["encoding"] == "cp1252"
