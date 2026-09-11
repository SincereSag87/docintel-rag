from hashlib import sha256
from pathlib import Path
from typing import Any

from app.documents.models import Document, SourceType


def document_id_from_file(path: Path) -> str:
    digest = sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def build_file_document(path: Path, text: str, metadata: dict[str, Any]) -> Document:
    return Document(
        id=document_id_from_file(path),
        filename=path.name,
        source_type=SourceType.FILE,
        text=text,
        metadata=metadata,
    )
