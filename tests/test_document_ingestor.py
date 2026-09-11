from pathlib import Path

from app.documents.models import Document, SourceType
from app.ingestion.base import DocumentLoader
from app.ingestion.document_ingestor import DocumentIngestor
from app.ingestion.models import DocumentFileType


class FakeLoader(DocumentLoader):
    def __init__(self) -> None:
        self.loaded_path: Path | None = None

    def load(self, path: Path) -> Document:
        self.loaded_path = path
        return Document(
            id="sha256:test",
            filename=path.name,
            source_type=SourceType.FILE,
            text="Alpha   beta\r\n\r\nGamma",
            metadata={"file_type": "txt"},
        )


def test_document_ingestor_selects_correct_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("placeholder", encoding="utf-8")
    loader = FakeLoader()

    document = DocumentIngestor(loaders={DocumentFileType.TXT: loader}).ingest(path)

    assert loader.loaded_path == path
    assert document.text == "Alpha beta\n\nGamma"
    assert document.metadata["document_type"] == "txt"
    assert document.metadata["normalized"] is True
