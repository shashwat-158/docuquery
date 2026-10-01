from pydantic import BaseModel, Field

from app import config


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


class Source(BaseModel):
    id: int
    filename: str
    page: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int