from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Configuration
    API_TITLE: str = "Legal Case Search API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
    ]
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = "legal_cases"
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "1536"))
    
    # Embedding Configuration 
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    USE_OPENAI_EMBEDDINGS: bool = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true"
    
    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "docx", "txt", "png", "jpg", "jpeg", "tiff", "html", "eml"]
    UPLOAD_DIR: str = "/tmp/uploads"
    
    # Text Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Search Configuration
    SEARCH_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.4
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
