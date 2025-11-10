from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import List
import os
import uuid
import logging
from datetime import datetime
import asyncio
import time

from app.config import settings
from app.services.parser_service import ParserService
from app.services.embed_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.utils.text_splitter import TextSplitter
from app.schemas.document import UploadResponse, DocumentListResponse, ListedDocument

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
embed_service = EmbeddingService()
qdrant_service = QdrantService()
text_splitter = TextSplitter(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    user_id: str = Form(..., description="User UUID to scope uploaded documents")
):
    """Upload and process legal documents"""
    results = []
    
    for file in files:
        try:
            # Validate file type
            file_ext = file.filename.split('.')[-1].lower()
            if file_ext not in settings.ALLOWED_FILE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type .{file_ext} not allowed"
                )
            
            # Create upload directory
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            
            # Save file
            file_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
            t0 = time.perf_counter()
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_size_kb = len(content) // 1024
            logger.info(f"Saved '{file.filename}' ({file_size_kb} KB) to {file_path}")
            if file_size_kb > settings.MAX_FILE_SIZE_MB * 1024:
                logger.warning(
                    f"File '{file.filename}' exceeds MAX_FILE_SIZE_MB={settings.MAX_FILE_SIZE_MB}; processing may be slow"
                )
            
            # Parse file
            logger.info(f"Starting parse for '{file.filename}'")
            text = await asyncio.to_thread(ParserService.parse_file, file_path)
            logger.info(f"Finished parse for '{file.filename}' in {time.perf_counter()-t0:.2f}s")
            if not text:
                raise HTTPException(status_code=400, detail="Could not extract text from file")
            
            # Split text into chunks
            t1 = time.perf_counter()
            chunk_dicts = text_splitter.split(text)
            chunk_texts = [c["content"] for c in chunk_dicts]
            logger.info(f"Split into {len(chunk_texts)} chunks in {time.perf_counter()-t1:.2f}s")
            
            # Generate embeddings
            t2 = time.perf_counter()
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")
            try:
                # Fail fast if embeddings take too long, so user sees a clear error
                embeddings = await asyncio.wait_for(
                    asyncio.to_thread(embed_service.embed_batch, chunk_texts),
                    timeout=180
                )
            except asyncio.TimeoutError:
                logger.error("Embedding generation timed out after 180s")
                raise HTTPException(status_code=504, detail="Embedding generation timed out")
            logger.info(f"Generated embeddings in {time.perf_counter()-t2:.2f}s")
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Store in Qdrant
            metadata = {
                "filename": file.filename,
                "file_type": file_ext,
                "upload_date": datetime.utcnow().isoformat(),
                "file_size_kb": file_size_kb,
            }
            
            t3 = time.perf_counter()
            logger.info("Initializing Qdrant client")
            await qdrant_service.initialize()
            logger.info(f"Qdrant initialized in {time.perf_counter()-t3:.2f}s")
            logger.info("Upserting embeddings to Qdrant")
            chunk_ids = await qdrant_service.store_embeddings(
                chunk_texts,
                embeddings,
                doc_id,
                metadata,
                user_id=user_id,
            )
            logger.info(f"Upserted {len(chunk_ids)} points to Qdrant in {time.perf_counter()-t3:.2f}s total")
            
            # Clean up temporary file
            os.remove(file_path)
            
            results.append(
                UploadResponse(
                    doc_id=doc_id,
                    filename=file.filename,
                    chunks_created=len(chunk_ids),
                    message=f"Successfully uploaded {file.filename}",
                    status="success"
                )
            )
            
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            results.append(
                UploadResponse(
                    doc_id="",
                    filename=file.filename,
                    chunks_created=0,
                    message=f"Failed to upload: {str(e)}",
                    status="error"
                )
            )
    
    return {"uploads": [r.dict() for r in results]}

@router.get("/list_documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        await qdrant_service.initialize()
        stats = await qdrant_service.get_collection_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def get_user_documents(user_id: str = Query(..., description="User UUID")):
    """List uploaded documents for a given user_id"""
    try:
        await qdrant_service.initialize()
        docs = await qdrant_service.list_documents_by_user(user_id)

        # Map to schema models
        listed = []
        for d in docs:
            # Convert upload_date to datetime when possible
            upload_dt = None
            try:
                upload_dt = datetime.fromisoformat(d.get("upload_date", ""))
            except Exception:
                upload_dt = datetime.utcnow()
            listed.append(
                ListedDocument(
                    doc_id=d.get("doc_id", ""),
                    filename=d.get("filename", "unknown"),
                    file_type=d.get("file_type", ""),
                    upload_date=upload_dt,
                    file_size_kb=int(d.get("file_size_kb", 0)),
                    num_chunks=int(d.get("num_chunks", 0)),
                )
            )

        return DocumentListResponse(documents=listed, total_count=len(listed))
    except Exception as e:
        logger.error(f"Error getting documents for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{doc_id}")
async def delete_user_document(doc_id: str, user_id: str = Query(..., description="User UUID")):
    """Delete a document (all its chunks) for a given user_id"""
    try:
        await qdrant_service.initialize()
        ok = await qdrant_service.delete_document(doc_id, user_id=user_id)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        return {"status": "success", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
