from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    """Search query request"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    threshold: float = Field(0.4, ge=0.0, le=1.0, description="Similarity threshold")
    sort_by: str = Field("relevance", description="Sort results by 'relevance' or 'source'")
    rephrase: bool = Field(True, description="Rephrase query before embedding")
    rerank: bool = Field(True, description="LLM-guided reranking of retrieved chunks")
    user_id: Optional[str] = Field(None, description="User UUID to scope search collection")

class SearchResult(BaseModel):
    """UI-friendly search result card"""
    doc_id: str
    source: str
    similarity_score: float
    title: str
    summary: str

class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

class SearchSuggestionsResponse(BaseModel):
    """AI suggestions for search queries"""
    suggestions: List[str] = Field(default_factory=list)
