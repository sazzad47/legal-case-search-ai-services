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

        # Optional rephrase: disable when threshold is very high (favor exact phrasing)
        threshold_val = float(getattr(payload, "threshold", 0.0) or 0.0)
        should_rephrase = getattr(payload, "rephrase", True) and threshold_val < 0.95
        effective_query = rag_service.rephrase_query(payload.query) if should_rephrase else payload.query

        query_embedding = embed_service.embed(effective_query)
        await qdrant_service.initialize()
        # Clamp threshold to realistic cosine similarity range (< 1.0)
        effective_threshold = min(max(threshold_val, 0.0), 0.999)
        results = await qdrant_service.search(
            query_embedding,
            payload.top_k,
            effective_threshold,
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
            # No plain text summary needed in search results
            summary = ""
            search_results.append(
                SearchResult(
                    doc_id=doc_id,
                    source=g.get("source", ""),
                    similarity_score=g.get("best_score", 0.0),
                    title=card.get("title", "Untitled"),
                    summary=summary,
                    short_description_html=card.get("short_description_html"),
                    summary_html=card.get("summary_html"),
                )
            )

        # Fallback to doc_id as key when source is missing
        dedup_map = {}
        for r in search_results:
            key = (r.source or "").strip().lower() or r.doc_id
            existing = dedup_map.get(key)
            if not existing or float(r.similarity_score) > float(existing.similarity_score):
                dedup_map[key] = r
        search_results = list(dedup_map.values())

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
async def search_suggestions(user_id: str, q: str = "", limit: int = 6):
    """Return AI-powered query suggestions specific to the given user_id.

    Tailors suggestions to the user's uploaded documents. No generic fallbacks.
    """
    try:
        await qdrant_service.initialize()
        docs = await qdrant_service.list_documents_by_user(user_id)

        # Build lightweight context from user's documents
        doc_lines = []
        for d in docs[:10]:  # cap for prompt brevity
            name = d.get("filename", "unknown")
            ftype = d.get("file_type", "")
            num = d.get("num_chunks", 0)
            doc_lines.append(f"- {name} ({ftype}, chunks: {num})")
        documents_context = "\n".join(doc_lines)

        suggestions = rag_service.suggest_queries(q, limit, documents_context=documents_context, user_id=user_id)
        return SearchSuggestionsResponse(suggestions=suggestions)
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        # No fallback suggestions; return empty list
        return SearchSuggestionsResponse(suggestions=[])

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
