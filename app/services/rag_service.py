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

    def suggest_queries(self, seed: str = "", limit: int = 6, documents_context: str | None = None, user_id: str | None = None) -> list:
        """Suggest helpful search queries tailored to a specific user's uploaded documents.

        - Requires OpenAI; returns an empty list when unavailable or on error.
        - Uses the provided documents_context to guide query generation.
        """
        seed = (seed or "").strip()

        if not self.use_openai:
            # No fallbacks; honor requirement for user-specific suggestions only
            return []

        try:
            instruction = (
                "You are assisting with search over a specific user's uploaded legal documents. "
                "Propose concise, diverse search queries that would be most effective for THIS corpus. "
                "Use only the provided document context; avoid generic topics unrelated to it. "
                "Return ONLY a JSON array of strings."
            )
            user_prompt = (
                (f"Seed: {seed}\n" if seed else "") +
                (f"User ID: {user_id}\n" if user_id else "") +
                (f"Documents:\n{documents_context}" if documents_context else "No document context provided.")
            )
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
            suggestions: list = []
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and parsed:
                    suggestions = parsed
            except Exception:
                # No fallback suggestions
                return []
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
            logger.warning(f"Suggest queries failed; returning empty list. Error: {e}")
            return []

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
            empty_html = "<p>No relevant content found.</p>"
            return {
                "title": "No title",
                "summary": "No relevant content found.",
                "short_description_html": empty_html,
                "summary_html": empty_html,
            }

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
            # Heuristic fallback producing paragraph-only HTML with basic highlights
            sentences = [s.strip() for s in re.split(r"[\.!?]\s+", text_context) if s.strip()]
            title = detected_title or (sentences[0][:80] if sentences else (chunks[0][:80]).strip())

            # Pick an issue/context sentence
            issue_sentence = None
            for s in sentences:
                if any(k in s.lower() for k in ["case", "dispute", "conflict", "commerce", "permit", "law", "bridge", "contract", "constitution"]):
                    issue_sentence = s
                    break

            # Pick a holding/rule sentence
            holding_sentence = None
            for s in sentences:
                if any(k in s.lower() for k in ["held", "ruled", "decided", "court", "declared", "struck", "void", "valid", "unconstitutional", "supremacy clause", "interstate commerce"]):
                    holding_sentence = s
                    break

            fallback_summary_text = self._simple_summarize(query, chunks)
            parts = []
            if issue_sentence:
                parts.append(issue_sentence)
            if holding_sentence and holding_sentence != issue_sentence:
                parts.append(holding_sentence)
            if not parts:
                parts.append(fallback_summary_text)

            summary_text = ". ".join([p.strip() for p in parts])

            def mark_highlights(text: str, q: str) -> str:
                try:
                    if not q:
                        return text
                    # highlight key tokens from query
                    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", q) if len(t) > 2]
                    highlighted = text
                    for t in set(tokens):
                        highlighted = re.sub(rf"(?i)\b{re.escape(t)}\b", f"<mark>{t}</mark>", highlighted)
                    return highlighted
                except Exception:
                    return text

            short_description_html = f"<p>{mark_highlights(issue_sentence or (sentences[0] if sentences else fallback_summary_text), query)}</p>"
            summary_html = f"<p>{mark_highlights(summary_text, query)}</p>"

            return {
                "title": title.strip(),
                "summary": summary_text.strip(),
                "short_description_html": short_description_html,
                "summary_html": summary_html,
            }

        try:
            instruction = (
                "You are a legal research assistant. Build a UI card in JSON. "
                "Return ONLY JSON with keys: 'title', 'short_description_html', 'summary_html'. "
                "If a case name is provided, use it verbatim in 'title'. "
                "Title: 6–12 words, include year if available. "
                "short_description_html: ONE crisp sentence (<p>...</p>) summarizing the issue/context, 20–40 words. "
                "summary_html: ONE compact paragraph (<p>...</p>) focusing on highlights and context relevant to the query. "
                "Include highlights with <mark> for key entities, year, doctrines, and holdings. "
                "No lists or bullets. Use plain legal English, no citations or disclaimers, and valid HTML only."
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
            import json, re
            try:
                parsed = json.loads(content)
                title = (parsed.get("title") or detected_title or "Untitled").strip()
                short_description_html = (parsed.get("short_description_html") or "").strip()
                summary_html = (parsed.get("summary_html") or "").strip()
                # sanitize: remove list tags, ensure paragraph wrapper
                def ensure_paragraph(html: str) -> str:
                    try:
                        clean = re.sub(r"</?ul[^>]*>|</?li[^>]*>", " ", html or "")
                        clean = clean.strip()
                        if not clean:
                            return "<p></p>"
                        if not clean.lower().startswith("<p>"):
                            clean = f"<p>{clean}</p>"
                        return clean
                    except Exception:
                        return f"<p>{html or ''}</p>"
                short_description_html = ensure_paragraph(short_description_html)
                summary_html = ensure_paragraph(summary_html)
                # Derive a plain-text summary for backward compatibility
                summary_text = re.sub(r"<[^>]+>", " ", summary_html).replace("\n", " ").strip()
                return {
                    "title": title,
                    "summary": summary_text,
                    "short_description_html": short_description_html,
                    "summary_html": summary_html,
                }
            except Exception:
                # Fallback to heuristic
                title = detected_title or (chunks[0][:80]).strip()
                summary_text = self._simple_summarize(query, chunks)
                short_description_html = f"<p>{summary_text}</p>"
                summary_html = f"<p>{summary_text}</p>"
                return {
                    "title": title,
                    "summary": summary_text,
                    "short_description_html": short_description_html,
                    "summary_html": summary_html,
                }
        except Exception as e:
            logger.warning(f"Card generation failed; using fallback: {e}")
            title = detected_title or (chunks[0][:80]).strip()
            summary = self._simple_summarize(query, chunks)
            return {"title": title, "summary": summary}
