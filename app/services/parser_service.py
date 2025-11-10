from typing import List, Tuple
import logging
import os
from app.utils.logger import setup_logger
from app.utils.cleaner import TextCleaner

logger = setup_logger(__name__)

class ParserService:
    """Parses various file formats"""
    
    @staticmethod
    def parse_txt(file_path: str) -> str:
        """Parse text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error parsing TXT file: {e}")
            return ""
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Parse PDF file"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages; starting extraction")
                for idx, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    text += page_text
                    if idx % 10 == 0 or idx == total_pages:
                        logger.info(f"Extracted text from {idx}/{total_pages} pages")
            return text
        except ImportError:
            logger.error("pdfplumber not installed")
            return ""
        except Exception as e:
            logger.error(f"Error parsing PDF file: {e}")
            return ""
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Parse DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except ImportError:
            logger.error("python-docx not installed")
            return ""
        except Exception as e:
            logger.error(f"Error parsing DOCX file: {e}")
            return ""
    
    @staticmethod
    def parse_html(file_path: str) -> str:
        """Parse HTML file"""
        try:
            from html.parser import HTMLParser
            class MLStripper(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.reset()
                    self.strict = False
                    self.convert_charrefs = True
                    self.text = []
                def handle_data(self, d):
                    self.text.append(d)
                def get_data(self):
                    return ''.join(self.text)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            stripper = MLStripper()
            stripper.feed(html_content)
            return stripper.get_data()
        except Exception as e:
            logger.error(f"Error parsing HTML file: {e}")
            return ""
    
    @staticmethod
    def parse_image(file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(file_path)
            text = "\n".join([result[1] for result in results])
            return text
        except ImportError:
            logger.error("easyocr not installed")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    @staticmethod
    def parse_file(file_path: str) -> str:
        """Parse file based on extension"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        logger.info(f"Parsing file with extension: {ext}")
        
        if ext == '.pdf':
            text = ParserService.parse_pdf(file_path)
        elif ext == '.docx':
            text = ParserService.parse_docx(file_path)
        elif ext == '.html':
            text = ParserService.parse_html(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            text = ParserService.parse_image(file_path)
        elif ext == '.txt':
            text = ParserService.parse_txt(file_path)
        elif ext == '.eml':
            # Simple EML parsing
            text = ParserService.parse_txt(file_path)
        else:
            text = ParserService.parse_txt(file_path)  # Fallback
        
        # Clean text
        text = TextCleaner.clean(text)
        return text
