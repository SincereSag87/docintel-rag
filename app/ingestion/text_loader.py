from pathlib import Path

from app.documents.models import Document
from app.ingestion.base import DocumentExtractionError, DocumentLoader, EmptyDocumentError
from app.ingestion.utils import build_file_document


class TextLoader(DocumentLoader):
    encodings = ("utf-8-sig", "utf-8", "cp1252")

    def load(self, path: Path) -> Document:
        text: str | None = None
        errors: list[str] = []

        for encoding in self.encodings:
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc.reason}")
            except OSError as exc:
                raise DocumentExtractionError(f"Could not read text file '{path.name}'.") from exc

        if text is None:
            detail = "; ".join(errors) or "unknown encoding error"
            raise DocumentExtractionError(f"Could not decode text file '{path.name}': {detail}.")

        if not text.strip():
            raise EmptyDocumentError(f"Text file '{path.name}' is empty.")

        return build_file_document(
            path=path,
            text=text,
            metadata={
                "file_type": "txt",
                "encoding": encoding,
                "byte_size": path.stat().st_size,
            },
        )
