from pathlib import Path
from typing import Any

import pymupdf

from app.documents.models import Document
from app.ingestion.base import DocumentExtractionError, DocumentLoader
from app.ingestion.utils import build_file_document

MIN_EXTRACTED_TEXT_CHARS = 20


class PDFLoader(DocumentLoader):
    def load(self, path: Path) -> Document:
        try:
            pdf = pymupdf.open(path)
        except Exception as exc:
            raise DocumentExtractionError(f"Could not read PDF file '{path.name}'.") from exc

        try:
            page_texts = []
            for index, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    page_texts.append(f"[Page {index}]\n{text}")

            text = "\n\n".join(page_texts)
            if len(text.strip()) < MIN_EXTRACTED_TEXT_CHARS:
                raise DocumentExtractionError(
                    "No readable text could be extracted. "
                    "The PDF may be scanned or image-based."
                )

            metadata = self._metadata(pdf)
            metadata["file_type"] = "pdf"
            metadata["page_count"] = pdf.page_count
            return build_file_document(path=path, text=text, metadata=metadata)
        finally:
            pdf.close()

    @staticmethod
    def _metadata(pdf: pymupdf.Document) -> dict[str, Any]:
        raw_metadata = pdf.metadata or {}
        return {
            "author": raw_metadata.get("author") or None,
            "title": raw_metadata.get("title") or None,
            "subject": raw_metadata.get("subject") or None,
        }
