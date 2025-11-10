from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routers import upload, search, ask, health
from app.config import settings
from app.services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
qdrant_service = QdrantService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    try:
        await qdrant_service.initialize()
        logger.info("Qdrant service initialized successfully")
    except Exception as e:
        # Do not crash app on startup; Qdrant may come online later.
        logger.error(f"Failed to initialize Qdrant on startup: {e}")
    yield
    logger.info("Shutting down application")

app = FastAPI(
    title="Legal Case Search API",
    description="FastAPI backend for RAG-based legal case search",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(search.router, tags=["Search"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Legal Case Search API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
