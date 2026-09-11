from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SourceType(StrEnum):
    FILE = "file"
    URL = "url"
    TEXT = "text"
    UNKNOWN = "unknown"


def new_id() -> str:
    return str(uuid4())


class Document(BaseModel):
    id: str = Field(default_factory=new_id)
    filename: str
    source_type: SourceType = SourceType.FILE
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=new_id)
    document_id: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    chunk: DocumentChunk
    embedding: list[float] = Field(min_length=1)


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int = Field(ge=0)
    excerpt: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
