from pathlib import Path

from app.ingestion.document_ingestor import DocumentIngestor


def test_same_contents_produce_same_document_id(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Same policy text", encoding="utf-8")
    second.write_text("Same policy text", encoding="utf-8")

    ingestor = DocumentIngestor()

    assert ingestor.ingest(first).id == ingestor.ingest(second).id


def test_different_contents_produce_different_document_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("PTO policy", encoding="utf-8")
    second.write_text("Expense policy", encoding="utf-8")

    ingestor = DocumentIngestor()

    assert ingestor.ingest(first).id != ingestor.ingest(second).id
