import numpy as np
from typing import List
import logging
import time
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    """Generates embeddings for text"""
    
    def __init__(self):
        # OpenAI-only embeddings per user request
        self.use_openai = settings.USE_OPENAI_EMBEDDINGS and bool(settings.OPENAI_API_KEY)
        self.client = None
        if self.use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("Using OpenAI embeddings")
            except Exception as e:
                logger.warning(f"OpenAI client init failed, falling back to simple embeddings: {e}")
                self.use_openai = False
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not text or not text.strip():
            return [0.0] * settings.VECTOR_SIZE
        
        try:
            if self.use_openai and self.client:
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embedding = response.data[0].embedding
                # Normalize to configured vector size
                if len(embedding) < settings.VECTOR_SIZE:
                    embedding.extend([0.0] * (settings.VECTOR_SIZE - len(embedding)))
                else:
                    embedding = embedding[:settings.VECTOR_SIZE]
                return embedding
            else:
                return self._simple_embed(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return self._simple_embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with progress logging and batching"""
        if not texts:
            return []

        # OpenAI batch path for efficiency, with retries and progress logs
        if self.use_openai and self.client:
            batch_size = 64
            max_retries = 3
            total = len(texts)
            results: List[List[float]] = []
            t_start = time.perf_counter()
            logger.info(f"Embedding {total} chunks via OpenAI in batches of {batch_size}")

            for start_idx in range(0, total, batch_size):
                batch = texts[start_idx:start_idx + batch_size]
                attempt = 0
                batch_t0 = time.perf_counter()
                while attempt < max_retries:
                    try:
                        response = self.client.embeddings.create(
                            input=batch,
                            model="text-embedding-3-small",
                        )
                        # response.data is a list aligned with input order
                        for item in response.data:
                            emb = item.embedding
                            if len(emb) < settings.VECTOR_SIZE:
                                emb.extend([0.0] * (settings.VECTOR_SIZE - len(emb)))
                            else:
                                emb = emb[:settings.VECTOR_SIZE]
                            results.append(emb)
                        logger.info(
                            f"Embedded batch {start_idx//batch_size + 1} "
                            f"({len(batch)} items) in {time.perf_counter()-batch_t0:.2f}s; "
                            f"progress {len(results)}/{total}"
                        )
                        break
                    except Exception as e:
                        attempt += 1
                        logger.warning(
                            f"OpenAI embeddings batch failed (attempt {attempt}/{max_retries}): {e}"
                        )
                        time.sleep(min(2 ** attempt, 8))
                else:
                    # Fallback for this batch if all retries failed
                    logger.error(
                        f"OpenAI embeddings failed for batch starting at {start_idx}; "
                        "falling back to simple embeddings for this batch"
                    )
                    for text in batch:
                        results.append(self._simple_embed(text))

            logger.info(f"Finished embedding {total} chunks in {time.perf_counter()-t_start:.2f}s")
            return results

        # Fallback path: per-item simple embeddings with progress logs
        embeddings = []
        t_start = time.perf_counter()
        for i, text in enumerate(texts, 1):
            embeddings.append(self._simple_embed(text))
            if i % 50 == 0 or i == len(texts):
                logger.info(f"Simple-embedded {i}/{len(texts)} chunks")
        logger.info(f"Finished simple embeddings in {time.perf_counter()-t_start:.2f}s")
        return embeddings
    
    def _simple_embed(self, text: str) -> List[float]:
        """Fallback simple embedding based on text hash"""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(settings.VECTOR_SIZE).tolist()
