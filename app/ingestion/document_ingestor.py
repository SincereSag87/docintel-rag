from pathlib import Path

from app.documents.models import Document
from app.ingestion.base import (
    DocumentLoader,
    DocumentNotFoundError,
    EmptyDocumentError,
)
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.models import DocumentFileType, detect_document_type
from app.ingestion.normalizer import normalize_text
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_loader import TextLoader


class DocumentIngestor:
    def __init__(self, loaders: dict[DocumentFileType, DocumentLoader] | None = None) -> None:
        self.loaders = loaders or {
            DocumentFileType.PDF: PDFLoader(),
            DocumentFileType.DOCX: DocxLoader(),
            DocumentFileType.TXT: TextLoader(),
        }

    def ingest(self, path: str | Path) -> Document:
        document_path = Path(path).expanduser()
        self._validate_path(document_path)

        document_type = detect_document_type(document_path)
        loader = self.loaders[document_type]
        document = loader.load(document_path)
        normalized_text = normalize_text(document.text)
        if not normalized_text:
            raise EmptyDocumentError(f"Document '{document_path.name}' contains no readable text.")

        metadata = {
            **document.metadata,
            "document_type": document_type.value,
            "normalized": True,
        }
        return document.model_copy(update={"text": normalized_text, "metadata": metadata})

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.exists():
            raise DocumentNotFoundError(f"Document path does not exist: {path}")
        if not path.is_file():
            raise DocumentNotFoundError(f"Document path is not a file: {path}")
