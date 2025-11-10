from pydantic import BaseModel, Field
from typing import List, Optional

class RAGRequest(BaseModel):
    """RAG query request"""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    include_sources: bool = Field(True, description="Include source information")
    user_id: Optional[str] = Field(None, description="User UUID to scope collection")

class RAGResponse(BaseModel):
    """RAG response with answer and sources"""
    question: str
    answer: str
    sources: List[dict] = Field(default_factory=list)
    processing_time_ms: float
