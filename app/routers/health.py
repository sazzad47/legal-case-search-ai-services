from fastapi import APIRouter
from app.services.qdrant_service import QdrantService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Legal Case Search API"
    }

@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        qdrant = QdrantService()
        await qdrant.initialize()
        stats = await qdrant.get_collection_stats()
        return {
            "status": "success",
            "database_stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
