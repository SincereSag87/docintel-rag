from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and .env."""

    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        alias="OLLAMA_BASE_URL",
    )
    default_model: str = Field(default="llama3.2", alias="DEFAULT_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE", gt=0)
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP", ge=0)
    top_k: int = Field(default=5, alias="TOP_K", gt=0)
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="docintel", alias="CHROMA_COLLECTION", min_length=1)

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE.")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
