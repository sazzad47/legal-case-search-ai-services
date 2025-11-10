from fastapi import APIRouter, HTTPException
import logging
from time import time
from typing import Optional

from app.services.embed_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.rag_service import RAGService
from app.schemas.search import SearchRequest, SearchResponse, SearchResult, SearchSuggestionsResponse

logger = logging.getLogger(__name__)
router = APIRouter()

embed_service = EmbeddingService()
qdrant_service = QdrantService()
rag_service = RAGService()

@router.post("/search")
async def search_post(payload: SearchRequest):
    """POST variant aligning with UI expectations.

    Accepts JSON body and runs the same best-practice flow.
    """
    try:
        start_time = time()

        # Optional rephrase
        effective_query = rag_service.rephrase_query(payload.query) if getattr(payload, "rephrase", True) else payload.query

        query_embedding = embed_service.embed(effective_query)
        await qdrant_service.initialize()
        results = await qdrant_service.search(
            query_embedding,
            payload.top_k,
            payload.threshold,
            user_id=getattr(payload, "user_id", None),
        )

        # Optional rerank of chunks
        use_rerank = getattr(payload, "rerank", True)
        if use_rerank:
            results = rag_service.rerank_results(payload.query, results, payload.top_k)
        else:
            results = sorted(results, key=lambda r: r.get("similarity_score", 0.0), reverse=True)[: payload.top_k]

        # Group chunks by document to produce UI-friendly cards
        grouped = {}
        for r in results:
            doc_id = r.get("doc_id")
            if not doc_id:
                # Skip if missing
                continue
            g = grouped.setdefault(doc_id, {"chunks": [], "best_score": -1.0, "source": r.get("source", "")})
            g["chunks"].append(r.get("content", ""))
            g["best_score"] = max(g["best_score"], float(r.get("similarity_score", 0.0)))

        # Build cards per document
        search_results = []
        for doc_id, g in grouped.items():
            card = rag_service.build_result_card(payload.query, g["chunks"])
            summary = card.get("summary", "")
            # Truncate summary for UI preview
            max_len = 240
            if len(summary) > max_len:
                summary = summary[:max_len].rstrip() + "…"
            search_results.append(
                SearchResult(
                    doc_id=doc_id,
                    source=g.get("source", ""),
                    similarity_score=g.get("best_score", 0.0),
                    title=card.get("title", "Untitled"),
                    summary=summary,
                )
            )

        # Sort results by relevance or source
        sort_by = getattr(payload, "sort_by", "relevance")
        if sort_by == "source":
            search_results = sorted(search_results, key=lambda r: (r.source or "").lower())
        else:
            search_results = sorted(search_results, key=lambda r: r.similarity_score, reverse=True)

        # Clip to top_k documents
        search_results = search_results[: payload.top_k]
        search_time_ms = (time() - start_time) * 1000

        return SearchResponse(
            query=effective_query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms,
        )
    except Exception as e:
        logger.error(f"Search POST error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/suggestions")
async def search_suggestions(q: str = "", limit: int = 6):
    """Return AI-powered query suggestions.

    - If `q` is provided, tailor suggestions to that seed.
    - Falls back to curated suggestions when AI is unavailable.
    """
    try:
        suggestions = rag_service.suggest_queries(q, limit)
        return SearchSuggestionsResponse(suggestions=suggestions)
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        # Conservative fallback
        return SearchSuggestionsResponse(suggestions=[
            "breach of contract in retail",
            "landlord liability for tenant injury",
            "intellectual property disputes",
            "employment discrimination cases",
        ][:limit])

@router.get("/case/{doc_id}")
async def get_case_card(doc_id: str, user_id: Optional[str] = None, query: str = ""):
    """Generate a full case card (title + description) for a specific document.

    - Fetches chunks for the given doc_id from the user's collection
    - Uses AI to craft a concise, readable title and description
    """
    try:
        await qdrant_service.initialize()
        payloads = await qdrant_service.get_document_chunks(doc_id, user_id=user_id, limit=24)
        chunks = [p.get("content", "") for p in payloads]
        card = rag_service.build_result_card(query or "Summarize this case.", chunks)
        source = payloads[0].get("filename", "") if payloads else ""
        return {
            "doc_id": doc_id,
            "source": source,
            "title": card.get("title", "Untitled"),
            "description": card.get("summary", ""),
        }
    except Exception as e:
        logger.error(f"Case card error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
