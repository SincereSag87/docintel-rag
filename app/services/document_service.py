from pathlib import Path

from app.documents.models import Document
from app.ingestion.document_ingestor import DocumentIngestor
from app.ingestion.models import DocumentInspection


class DocumentService:
    def __init__(self, ingestor: DocumentIngestor | None = None) -> None:
        self.ingestor = ingestor or DocumentIngestor()

    def ingest_document(self, path: str | Path) -> Document:
        return self.ingestor.ingest(path)

    def inspect_document(self, path: str | Path) -> DocumentInspection:
        document = self.ingest_document(path)
        return self.summarize(document)

    @staticmethod
    def summarize(document: Document, preview_chars: int = 500) -> DocumentInspection:
        preview = document.text[:preview_chars].strip()
        if len(document.text) > preview_chars:
            preview = f"{preview}..."

        return DocumentInspection(
            document_id=document.id,
            filename=document.filename,
            source_type=str(document.source_type),
            characters=len(document.text),
            words=len(document.text.split()),
            metadata=document.metadata,
            preview=preview,
        )
