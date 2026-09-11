import hashlib
import re

from app.chunking.base import Chunker, ChunkingError
from app.core.config import Settings, get_settings
from app.documents.models import Document, DocumentChunk

PAGE_MARKER_PATTERN = re.compile(r"\[Page\s+(\d+)\]")


class RecursiveTextChunker(Chunker):
    """Character-oriented chunker using readable boundaries before hard splits."""

    boundaries = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        if self.chunk_size <= 0:
            raise ChunkingError("Chunk size must be greater than zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ChunkingError("Chunk overlap must be non-negative and smaller than chunk size.")

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.text.strip()
        if not text:
            raise ChunkingError("Cannot chunk an empty document.")

        parts = [part for part in self._split_text(text) if part.strip()]
        parts = self._apply_overlap(parts)

        chunks: list[DocumentChunk] = []
        for index, part in enumerate(parts):
            chunk_text = part.strip()
            if not chunk_text:
                continue
            metadata = self._chunk_metadata(document, chunk_text, index)
            chunks.append(
                DocumentChunk(
                    id=self._chunk_id(document.id, index, chunk_text),
                    document_id=document.id,
                    text=chunk_text,
                    chunk_index=index,
                    metadata=metadata,
                )
            )

        if not chunks:
            raise ChunkingError("Document did not produce any non-empty chunks.")
        return chunks

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        for boundary in self.boundaries:
            if boundary in text:
                pieces = self._split_with_boundary(text, boundary)
                if len(pieces) > 1:
                    return self._pack_pieces(pieces)

        return [
            text[index : index + self.chunk_size]
            for index in range(0, len(text), self.chunk_size)
        ]

    @staticmethod
    def _split_with_boundary(text: str, boundary: str) -> list[str]:
        raw_pieces = text.split(boundary)
        pieces: list[str] = []
        for index, piece in enumerate(raw_pieces):
            if not piece:
                continue
            suffix = boundary if index < len(raw_pieces) - 1 else ""
            pieces.append(f"{piece}{suffix}")
        return pieces

    def _pack_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_text(piece.strip()))
                continue

            candidate = f"{current}{piece}" if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = piece

        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for previous, current in zip(chunks[:-1], chunks[1:], strict=True):
            prefix = previous[-self.chunk_overlap :].strip()
            overlapped.append(f"{prefix}\n{current}" if prefix else current)
        return overlapped

    def _chunk_id(self, document_id: str, chunk_index: int, text: str) -> str:
        payload = f"{document_id}|{chunk_index}|{self.chunk_size}|{self.chunk_overlap}|{text}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{document_id}:chunk:{chunk_index}:{digest[:16]}"

    @staticmethod
    def _chunk_metadata(document: Document, text: str, chunk_index: int) -> dict[str, object]:
        metadata: dict[str, object] = {
            "document_id": document.id,
            "filename": document.filename,
            "source_type": str(document.source_type),
            "chunk_index": chunk_index,
        }
        for key, value in document.metadata.items():
            if isinstance(value, str | int | float | bool) or value is None:
                metadata[key] = value

        pages = [int(match) for match in PAGE_MARKER_PATTERN.findall(text)]
        if pages:
            metadata["page"] = pages[0]
        return metadata
