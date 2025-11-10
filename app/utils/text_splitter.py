from typing import List

class TextSplitter:
    """Splits text into overlapping chunks"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split(self, text: str) -> List[dict]:
        """
        Split text into chunks with overlap
        Returns list of dicts with 'content', 'start_char', 'end_char'
        """
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            return [{"content": text, "start_char": 0, "end_char": text_length}]
        
        # Guard against invalid overlap that could cause non-progress
        overlap = max(0, min(self.overlap, self.chunk_size - 1))
        start = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            chunks.append({
                "content": chunk_text,
                "start_char": start,
                "end_char": end
            })
            next_start = end - overlap
            # Ensure forward progress even in edge cases
            if next_start <= start:
                next_start = start + 1
            start = next_start
        
        return chunks

def chunk_documents(texts: List[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Convenience function to chunk multiple documents"""
    splitter = TextSplitter(chunk_size, overlap)
    all_chunks = []
    for text in texts:
        chunks = splitter.split(text)
        all_chunks.extend([c["content"] for c in chunks])
    return all_chunks
