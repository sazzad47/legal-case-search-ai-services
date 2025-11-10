import logging
from typing import List, Dict
from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__)

class RAGService:
    """Handles RAG (Retrieval-Augmented Generation) operations"""
    
    def __init__(self):
        self.use_openai = settings.OPENAI_API_KEY != ""
        if self.use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except ImportError:
                logger.warning("OpenAI not installed")
                self.use_openai = False
    
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer using LLM with context"""
        if not context_chunks:
            return "No relevant information found to answer your question."
        
        # Prepare context
        context = "\n\n".join([f"Document chunk:\n{chunk}" for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""You are a legal research assistant. Using the following document chunks as context, 
answer the question concisely and accurately. If the answer cannot be found in the context, say so.

Question: {question}

Context:
{context}

Answer:"""
        
        try:
            if self.use_openai:
                response = self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a legal research assistant. Answer questions about legal cases and law based on provided documents."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            else:
                return self._simple_summarize(question, context_chunks)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._simple_summarize(question, context_chunks)

    def rephrase_query(self, query: str) -> str:
        """Rephrase a user query to improve embedding-based retrieval.

        - Uses OpenAI when available to expand and clarify the query for semantic search
        - Falls back to a simple heuristic cleaner when OpenAI is not available
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return query

        if not self.use_openai:
            # Simple heuristic: lowercase, remove extra whitespace, expand common legal terms
            import re
            q = re.sub(r"\s+", " ", cleaned.lower())
            replacements = {
                "ip": "intellectual property",
                "contract breach": "breach of contract",
                "landlord": "landlord liability",
            }
            for k, v in replacements.items():
                q = q.replace(k, v)
            return q

        try:
            instruction = (
                "Rewrite the user's query to optimize for semantic vector retrieval in a legal case corpus. "
                "Clarify entities, expand synonyms (e.g., 'IP' -> 'intellectual property'), and keep it concise. "
                "Return only the rephrased query without commentary."
            )
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": cleaned},
            ]
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=128,
            )
            content = response.choices[0].message.content or cleaned
            return content.strip()
        except Exception as e:
            logger.warning(f"Query rephrase failed; using original. Error: {e}")
            return cleaned

    def suggest_queries(self, seed: str = "", limit: int = 6) -> list:
        """Suggest helpful search queries for legal research.

        - If OpenAI is available, generate suggestions tailored to the user's seed.
        - Otherwise, return curated and heuristic suggestions.
        """
        seed = (seed or "").strip()

        # Fallback curated suggestions
        base_suggestions = [
            "breach of contract in retail",
            "landlord liability for tenant injury",
            "intellectual property disputes",
            "employment discrimination cases",
            "negligence in construction projects",
            "consumer protection class actions",
        ]

        if not self.use_openai:
            # Simple heuristic expansion when seed is provided
            if seed:
                expansions = [
                    f"case law on {seed}",
                    f"precedents related to {seed}",
                    f"{seed} recent judgments",
                ]
                suggestions = expansions + base_suggestions
                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for s in suggestions:
                    if s not in seen:
                        seen.add(s)
                        deduped.append(s)
                return deduped[:limit]
            return base_suggestions[:limit]

        try:
            instruction = (
                "Propose concise, diverse search queries for a legal case corpus. "
                "Focus on topics, parties, liabilities, and common legal issues. "
                "Return ONLY a JSON array of strings."
            )
            user_prompt = seed if seed else "Generate general-purpose legal research queries."
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_prompt},
            ]
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=200,
            )
            content = response.choices[0].message.content
            import json
            suggestions = base_suggestions
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and parsed:
                    suggestions = parsed + base_suggestions
            except Exception:
                # Keep fallback suggestions
                pass
            # Deduplicate and clip
            seen = set()
            result = []
            for s in suggestions:
                s = (s or "").strip()
                if not s:
                    continue
                if s not in seen:
                    seen.add(s)
                    result.append(s)
            return result[:limit]
        except Exception as e:
            logger.warning(f"Suggest queries failed; using fallback. Error: {e}")
            return base_suggestions[:limit]

    def rerank_results(self, question: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Optionally rerank retrieved chunks using LLM guidance, inspired by other-project.

        Falls back to score-based ranking if OpenAI is unavailable.
        """
        if not results:
            return []

        # Fast path: if no OpenAI, just sort by similarity score
        if not self.use_openai:
            return sorted(results, key=lambda r: r.get("similarity_score", 0.0), reverse=True)[:top_k]

        try:
            # Build a compact list of chunks
            chunk_summaries = [
                {
                    "chunk_id": r.get("chunk_id"),
                    "doc_id": r.get("doc_id"),
                    "score": r.get("similarity_score", 0.0),
                    "preview": (r.get("content", "")[:400]).replace("\n", " "),
                }
                for r in results
            ]

            instruction = (
                "Given a user question and retrieved candidate chunks, return the best "
                f"{top_k} chunk_ids ordered by relevance to the question. "
                "Prefer legal specificity, citations, and direct relevance. Respond ONLY with a JSON list of chunk_id strings."
            )

            messages = [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": (
                        "Question:\n" + question + "\n\nCandidates:\n" + str(chunk_summaries)
                    ),
                },
            ]

            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=256,
            )
            content = response.choices[0].message.content

            # Parse JSON array of chunk_ids (best-effort)
            import json
            selected_ids: List[str] = []
            try:
                selected_ids = json.loads(content)
                if not isinstance(selected_ids, list):
                    selected_ids = []
            except Exception:
                # If parsing fails, fall back to score ordering
                return sorted(results, key=lambda r: r.get("similarity_score", 0.0), reverse=True)[:top_k]

            # Build map and return ordered subset
            by_id = {r.get("chunk_id"): r for r in results}
            ordered = [by_id[cid] for cid in selected_ids if cid in by_id]

            # If fewer than top_k, fill from score-based leftovers
            if len(ordered) < top_k:
                remaining = [r for r in sorted(results, key=lambda r: r.get("similarity_score", 0.0), reverse=True) if r not in ordered]
                ordered.extend(remaining[: max(0, top_k - len(ordered))])

            return ordered[:top_k]
        except Exception as e:
            logger.warning(f"Rerank failed; falling back to score-based ranking: {e}")
            return sorted(results, key=lambda r: r.get("similarity_score", 0.0), reverse=True)[:top_k]
    
    def _simple_summarize(self, question: str, chunks: List[str]) -> str:
        """Simple fallback summarization"""
        # Extract key sentences from chunks
        relevant_text = " ".join(chunks)
        sentences = relevant_text.split('.')
        
        # Find sentences containing question keywords
        keywords = question.lower().split()
        scored_sentences = []
        
        for sentence in sentences:
            score = sum(1 for keyword in keywords if keyword in sentence.lower())
            if score > 0:
                scored_sentences.append((sentence.strip(), score))
        
        # Return top 2 sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        summary = ". ".join([s[0] for s in scored_sentences[:2]])
        
        if summary:
            return summary + "."
        else:
            return "Based on the available documents, I found relevant information but cannot provide a specific answer."

    def build_result_card(self, query: str, chunks: List[str]) -> Dict[str, str]:
        """Generate a compact title and summary suitable for UI cards.

        - Tries to detect a case name (e.g., "X v. Y, 1824") from text
        - Uses OpenAI when available with a precise prompt for legal summaries
        - Falls back to a stronger heuristic summarizer otherwise
        """
        text_context = "\n\n".join(chunks[:6]) if chunks else ""

        if not text_context:
            return {"title": "No title", "summary": "No relevant content found."}

        # Attempt to detect a case name and year from the text
        import re
        detected_title = None
        try:
            case_pattern = r"([A-Z][A-Za-z\.&\s]+ v\. [A-Z][A-Za-z\.&\s]+)(?:[,\s]*(\(?\d{4}\)?))?"
            m = re.search(case_pattern, text_context)
            if m:
                name = m.group(1).strip()
                year = (m.group(2) or "").strip().strip("()")
                detected_title = f"{name}{(', ' + year) if year else ''}"
        except Exception:
            detected_title = None

        if not self.use_openai:
            # Heuristic fallback:
            # - Title: prefer detected case name; else first meaningful sentence
            # - Summary: two sentences -> issue/context + holding/rule
            sentences = [s.strip() for s in re.split(r"[\.!?]\s+", text_context) if s.strip()]
            title = detected_title or (sentences[0][:80] if sentences else (chunks[0][:80]).strip())

            # Pick an issue/context sentence
            issue_sentence = None
            for s in sentences:
                if any(k in s.lower() for k in ["case", "dispute", "conflict", "commerce", "permit", "law"]):
                    issue_sentence = s
                    break

            # Pick a holding/rule sentence
            holding_sentence = None
            for s in sentences:
                if any(k in s.lower() for k in ["held", "ruled", "decided", "court", "supremacy clause", "interstate commerce"]):
                    holding_sentence = s
                    break

            parts = []
            if issue_sentence:
                parts.append(issue_sentence)
            if holding_sentence and holding_sentence != issue_sentence:
                parts.append(holding_sentence)
            if not parts:
                parts.append(self._simple_summarize(query, chunks))

            summary = ". ".join([p.strip() for p in parts])
            return {"title": title.strip(), "summary": summary.strip()}

        try:
            instruction = (
                "You are a legal research assistant. Create a concise card for UI. "
                "Return ONLY a JSON object with keys 'title' and 'summary'. "
                "If a case name is provided, use it verbatim in 'title'. "
                "Title: 6–12 words, include year if available. "
                "Summary: EXACTLY 2 sentences under 240 characters total: "
                "(1) brief issue/context, (2) holding + controlling doctrine. "
                "Avoid quotes, disclaimers, and citation artifacts; use plain legal English."
            )
            user_payload = (
                (f"Detected case: {detected_title}\n" if detected_title else "") +
                f"Query: {query}\n\nText:\n{text_context}"
            )
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_payload},
            ]
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=220,
            )
            content = response.choices[0].message.content
            import json
            try:
                parsed = json.loads(content)
                title = (parsed.get("title") or detected_title or "Untitled").strip()
                summary = (parsed.get("summary") or "").strip()
                # Basic cleanup
                summary = summary.replace("\n", " ").strip()
                return {"title": title, "summary": summary}
            except Exception:
                # Fallback to heuristic
                title = detected_title or (chunks[0][:80]).strip()
                summary = self._simple_summarize(query, chunks)
                return {"title": title, "summary": summary}
        except Exception as e:
            logger.warning(f"Card generation failed; using fallback: {e}")
            title = detected_title or (chunks[0][:80]).strip()
            summary = self._simple_summarize(query, chunks)
            return {"title": title, "summary": summary}
