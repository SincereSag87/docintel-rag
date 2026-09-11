from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.ingestion.base import UnsupportedDocumentTypeError


class DocumentFileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class DocumentInspection(BaseModel):
    document_id: str
    filename: str
    source_type: str
    characters: int
    words: int
    metadata: dict[str, object]
    preview: str

    model_config = ConfigDict(frozen=True)


SUPPORTED_EXTENSIONS: dict[str, DocumentFileType] = {
    ".pdf": DocumentFileType.PDF,
    ".docx": DocumentFileType.DOCX,
    ".txt": DocumentFileType.TXT,
}


def detect_document_type(path: Path) -> DocumentFileType:
    extension = path.suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[extension]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type '{extension or '<none>'}'. "
            f"Supported extensions: {supported}."
        ) from exc
