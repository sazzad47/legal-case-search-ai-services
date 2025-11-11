from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)
from typing import List, Dict
import logging
import uuid
from app.config import settings
from app.utils.logger import setup_logger
from app.utils.retry import retry_async

logger = setup_logger(__name__)

class QdrantService:
    """Manages Qdrant vector database operations"""
    
    def __init__(self):
        self.client = None
        self.collection_prefix = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.VECTOR_SIZE
        self._initialized = False

    async def initialize(self):
        """Initialize connection to Qdrant"""
        try:
            if settings.QDRANT_API_KEY:
                self.client = AsyncQdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=30.0,
                )
            else:
                self.client = AsyncQdrantClient(url=settings.QDRANT_URL, timeout=30.0)
            
            # Note: collection creation moved to ensure_collection() per user

            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    async def store_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        doc_id: str,
        metadata: Dict,
        user_id: str | None = None,
        batch_size: int = 64,
    ) -> List[str]:
        """Store embeddings in Qdrant with metadata.
        Uses batched upserts and retries for robustness.
        """
        if not self.client or not self._initialized:
            raise RuntimeError("Qdrant client not initialized")
        
        # Ensure per-user collection exists
        collection_name = await self.ensure_collection(user_id)

        chunk_ids = []
        points: List[PointStruct] = []

        total_chunks = len(texts)
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)

            point = PointStruct(
                # Qdrant requires point IDs to be unsigned integers or UUID strings.
                # Use a deterministic UUIDv5 based on the chunk_id and pass as string.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": text,
                    "chunk_index": i,
                    "filename": metadata.get("filename", "unknown"),
                    "upload_date": metadata.get("upload_date", ""),
                    "file_type": metadata.get("file_type", ""),
                    "file_size_kb": metadata.get("file_size_kb", 0),
                    "num_chunks": total_chunks,
                    "user_id": user_id or "",
                },
            )
            points.append(point)

        try:
            # Upsert in batches with retries
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]

                async def _upsert():
                    return await self.client.upsert(collection_name=collection_name, points=batch)

                await retry_async(_upsert, retries=3)

            logger.info(f"Stored {len(points)} embeddings for document {doc_id}")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.4,
        user_id: str | None = None,
    ) -> List[Dict]:
        """Search for similar vectors in Qdrant"""
        if not self.client or not self._initialized:
            raise RuntimeError("Qdrant client not initialized")
        
        try:
            collection_name = await self.ensure_collection(user_id)
            async def _search():
                return await self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
                )

            results = await retry_async(_search, retries=2)
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "chunk_id": result.payload.get("chunk_id"),
                    "doc_id": result.payload.get("doc_id"),
                    "content": result.payload.get("content"),
                    "similarity_score": result.score,
                    "source": result.payload.get("filename"),
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            collection_info = await self.client.get_collection(self.collection_prefix)
            return {
                "collection_name": self.collection_prefix,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "distance_type": str(collection_info.config.params.distance),
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def delete_document(self, doc_id: str, user_id: str | None = None) -> bool:
        """Delete all chunks for a document, optionally scoped by user_id"""
        try:
            collection_name = await self.ensure_collection(user_id)
            # Build filter; per-user collection makes user_id optional
            conditions = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]

            async def _delete():
                return await self.client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(must=conditions)
                )
            await retry_async(_delete, retries=2)
            logger.info(f"Deleted all chunks for document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    async def list_documents_by_user(self, user_id: str, limit: int = 1000) -> List[Dict]:
        """List unique documents for a given user_id's collection with aggregated metadata."""
        if not self.client or not self._initialized:
            raise RuntimeError("Qdrant client not initialized")

        try:
            collection_name = await self.ensure_collection(user_id)
            docs: Dict[str, Dict] = {}
            next_offset = None

            while True:
                result = await self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=None,
                    limit=min(256, limit),
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )

                # Support both tuple (records, next_offset) and object with attributes
                points = []
                if isinstance(result, tuple) and len(result) == 2:
                    points, next_offset = result
                else:
                    points = getattr(result, "points", []) or getattr(result, "records", [])
                    # Handle different offset property names across client versions
                    next_offset = (
                        getattr(result, "next_page_offset", None)
                        if getattr(result, "next_page_offset", None) is not None
                        else getattr(result, "next_offset", None)
                    )

                for p in points:
                    payload = getattr(p, "payload", {}) or {}
                    doc_id = payload.get("doc_id")
                    if not doc_id:
                        continue
                    if doc_id not in docs:
                        docs[doc_id] = {
                            "doc_id": doc_id,
                            "filename": payload.get("filename", "unknown"),
                            "file_type": payload.get("file_type", ""),
                            "upload_date": payload.get("upload_date", ""),
                            "file_size_kb": payload.get("file_size_kb", 0),
                            "num_chunks": 0,
                        }
                    docs[doc_id]["num_chunks"] += 1
                if not next_offset:
                    break

            # Sort by upload_date desc when available
            doc_list = list(docs.values())
            try:
                doc_list.sort(key=lambda d: d.get("upload_date", ""), reverse=True)
            except Exception:
                pass
            return doc_list
        except Exception as e:
            logger.error(f"Failed to list documents for user {user_id}: {e}")
            return []
    def derive_collection_name(self, user_id: str | None) -> str:
        """Derive per-user collection name from prefix and user_id."""
        if not user_id:
            return self.collection_prefix
        # Sanitize user_id to be safe for Qdrant collection naming
        import re
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        return f"{self.collection_prefix}__{safe_id}"

    async def get_document_chunks(self, doc_id: str, user_id: str | None = None, limit: int = 12) -> List[Dict]:
        """Fetch a subset of chunks for a specific document.

        Returns a list of payload dicts containing at least 'content', 'chunk_index', and 'source' fields.
        """
        if not self.client or not self._initialized:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = await self.ensure_collection(user_id)

        try:
            result = await self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    ]
                ),
            )

            # Support tuple or object formats
            points = []
            if isinstance(result, tuple) and len(result) == 2:
                points, _ = result
            else:
                points = getattr(result, "points", []) or getattr(result, "records", [])

            payloads = []
            for p in points:
                payload = getattr(p, "payload", {}) or {}
                payloads.append(payload)

            # Sort by chunk_index if available
            try:
                payloads.sort(key=lambda x: int(x.get("chunk_index", 0)))
            except Exception:
                pass

            return payloads
        except Exception as e:
            logger.error(f"Failed to fetch chunks for doc {doc_id}: {e}")
            return []

    async def ensure_collection(self, user_id: str | None):
        """Ensure the collection for the given user exists with correct vector size and indexes."""
        if not self.client or not self._initialized:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = self.derive_collection_name(user_id)
        try:
            info = await self.client.get_collection(collection_name)
            # Validate vector size and recreate if mismatch
            current_size = None
            cfg = getattr(info, "config", None)
            if cfg and getattr(cfg, "params", None):
                params = cfg.params
                if hasattr(params, "vectors") and getattr(params.vectors, "size", None):
                    current_size = params.vectors.size
                elif hasattr(params, "vector_size"):
                    current_size = params.vector_size
            if current_size is None:
                logger.warning(f"Could not determine vector size for collection '{collection_name}'; skipping validation")
            elif int(current_size) != int(self.vector_size):
                logger.warning(
                    f"Qdrant collection '{collection_name}' vector size {current_size} != expected {self.vector_size}; recreating"
                )
                await self.client.delete_collection(collection_name)
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
        except Exception:
            logger.info(f"Creating collection '{collection_name}'")
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

        # Ensure payload indexes for faster filtering/deletes
        for field in ("doc_id", "chunk_id", "user_id"):
            try:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        return collection_name
