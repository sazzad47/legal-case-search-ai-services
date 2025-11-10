import re
from typing import Optional

class TextCleaner:
    """Cleans and normalizes text"""
    
    @staticmethod
    def clean(text: str) -> str:
        """Clean text for processing"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-$$$$\:]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def normalize_for_search(text: str) -> str:
        """Normalize text for search"""
        text = TextCleaner.clean(text)
        text = text.lower()
        return text
