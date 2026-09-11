from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(min_length=1)


class ChatResponse(BaseModel):
    model: str
    content: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)
