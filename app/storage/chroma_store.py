from pathlib import Path
from typing import Any

import chromadb

from app.core.config import Settings, get_settings
from app.documents.models import DocumentChunk, EmbeddedChunk
from app.storage.base import VectorSearchResult, VectorStore
from app.storage.models import VectorStoreStats


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB-backed vector store for embedded document chunks."""

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path or self.settings.chroma_path)
        self.collection_name = collection_name or self.settings.chroma_collection
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        self.collection.upsert(
            ids=[embedded.chunk.id for embedded in chunks],
            documents=[embedded.chunk.text for embedded in chunks],
            embeddings=[embedded.embedding for embedded in chunks],
            metadatas=[self._metadata_for_chroma(embedded.chunk.metadata) for embedded in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        if not query_embedding:
            return []

        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "embeddings", "metadatas", "distances"],
        )
        return self._map_query_results(result)

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def clear(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def stats(self) -> VectorStoreStats:
        return VectorStoreStats(
            collection=self.collection_name,
            count=self.collection.count(),
            path=str(self.path),
        )

    def close(self) -> None:
        """Release Chroma resources so temporary stores can be removed on Windows."""
        release_system = getattr(self.client, "_release_system", None)
        if callable(release_system):
            release_system(str(self.path))
        clear_cache = getattr(self.client, "clear_system_cache", None)
        if callable(clear_cache):
            clear_cache()

    @staticmethod
    def _metadata_for_chroma(metadata: dict[str, object]) -> dict[str, str | int | float | bool]:
        chroma_metadata: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str | int | float | bool):
                chroma_metadata[key] = value
        return chroma_metadata

    def _map_query_results(self, result: dict[str, Any]) -> list[VectorSearchResult]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0]

        mapped: list[VectorSearchResult] = []
        for chunk_id, text, metadata, distance, embedding in zip(
            ids,
            documents,
            metadatas,
            distances,
            embeddings,
            strict=False,
        ):
            metadata = metadata or {}
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=str(metadata.get("document_id", "")),
                text=text,
                chunk_index=int(metadata.get("chunk_index", 0)),
                metadata=metadata,
            )
            mapped.append(
                VectorSearchResult(
                    chunk=EmbeddedChunk(
                        chunk=chunk,
                        embedding=[float(value) for value in embedding],
                    ),
                    distance=float(distance),
                    metadata=metadata,
                )
            )
        return mapped
