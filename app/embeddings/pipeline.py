from app.documents.models import DocumentChunk, EmbeddedChunk
from app.embeddings.base import EmbeddingProvider, EmbeddingResponseError


class EmbeddingPipeline:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        vectors = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingResponseError("Embedding count does not match chunk count.")

        dimensions = {len(vector) for vector in vectors if vector}
        if len(dimensions) != 1 or len(dimensions) == 0:
            raise EmbeddingResponseError("Embedding vectors must have one consistent dimension.")

        return [
            EmbeddedChunk(chunk=chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
