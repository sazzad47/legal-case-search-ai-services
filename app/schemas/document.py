from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Document metadata"""
    filename: str
    file_type: str
    upload_date: datetime
    file_size_kb: int
    num_chunks: int
    source_url: Optional[str] = None

class Document(BaseModel):
    """Document with metadata"""
    doc_id: str = Field(..., description="Unique document ID")
    title: str
    content: str
    metadata: DocumentMetadata

class DocumentListResponse(BaseModel):
    """Response for listing documents"""
    documents: List["ListedDocument"]
    total_count: int

class UploadResponse(BaseModel):
    """Response for file upload"""
    doc_id: str
    filename: str
    chunks_created: int
    message: str
    status: str = "success"

class ChunkInfo(BaseModel):
    """Information about a text chunk"""
    chunk_id: str
    doc_id: str
    content: str
    start_char: int
    end_char: int

class ListedDocument(BaseModel):
    """A listed document entry with doc_id and metadata fields."""
    doc_id: str
    filename: str
    file_type: str
    upload_date: datetime
    file_size_kb: int
    num_chunks: int
    source_url: Optional[str] = None
