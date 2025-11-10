# Legal Case Search - FastAPI Backend

Production-ready FastAPI backend for RAG-based legal case search with Qdrant vector database.

## Features

- Multi-format file support (PDF, DOCX, TXT, Images with OCR, HTML, EML)
- Vector embeddings with OpenAI or open-source models
- Semantic search via Qdrant vector database
- RAG-based question answering with LLM integration
- Comprehensive error handling and logging

## Quick Start

### Prerequisites

- Python 3.11+
- Qdrant vector database (running or cloud instance)
- Optional: OpenAI API key for embeddings and LLM

### Installation

1. Clone repository and navigate to backend directory:
\`\`\`bash
cd backend
\`\`\`

2. Create virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Setup environment variables:
\`\`\`bash
cp .env.example .env
\`\`\`

Edit `.env` with your configuration:
\`\`\`
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your-key-here  # Optional
USE_OPENAI_EMBEDDINGS=true    # If using OpenAI
\`\`\`

### Running Qdrant (Docker)

\`\`\`bash
docker run -p 6333:6333 qdrant/qdrant
\`\`\`

Or for cloud: Use Qdrant Cloud and set QDRANT_URL and QDRANT_API_KEY.

### Running the Backend

\`\`\`bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
\`\`\`

API will be available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
- `GET /health` - Service health status
- `GET /stats` - Database statistics

### Document Management
- `POST /upload` - Upload legal documents
- `GET /documents?user_id=your_user_id` - List all documents for a user

### Search
- `GET /search?query=your_query&top_k=5&threshold=0.4` - Semantic search

### RAG Question Answering
- `POST /ask` - Ask questions with RAG

\`\`\`json
{
  "question": "What are the penalties for breach of contract?",
  "top_k": 5,
  "include_sources": true
}
\`\`\`

## Architecture

\`\`\`
app/
├── main.py                 # FastAPI application entry
├── config.py              # Configuration management
├── routers/               # API route handlers
│   ├── upload.py
│   ├── search.py
│   ├── ask.py
│   └── health.py
├── services/              # Business logic
│   ├── parser_service.py
│   ├── embed_service.py
│   ├── qdrant_service.py
│   └── rag_service.py
├── schemas/               # Pydantic models
│   ├── document.py
│   ├── search.py
│   └── rag.py
└── utils/                 # Utilities
    ├── text_splitter.py
    ├── cleaner.py
    └── logger.py
\`\`\`

## Integration with Next.js Frontend

The frontend calls the backend API endpoints:

\`\`\`typescript
// Upload file
const formData = new FormData();
formData.append("files", file);
await fetch("http://localhost:8000/upload", {
  method: "POST",
  body: formData
});

// Search
const response = await fetch(
  "http://localhost:8000/search?query=breach+of+contract&top_k=5"
);

// Ask question (RAG)
const response = await fetch("http://localhost:8000/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "What are common contract disputes?",
    top_k: 5,
    include_sources: true
  })
});
\`\`\`

## Configuration Options

See `.env.example` for all available options:

- **Embeddings**: Use OpenAI or sentence-transformers
- **Vector Size**: Default 384 (sentence-transformers), 1536 (OpenAI)
- **Chunk Size**: Default 500 characters with 50-char overlap
- **Search**: Default top 5 results with 0.4 similarity threshold

## Performance Tips

1. Use OpenAI embeddings for better quality but higher cost
2. Tune chunk size based on your documents (500-1000 chars typical)
3. Set appropriate similarity threshold (0.3-0.6 range)
4. Consider caching for repeated queries
5. Monitor Qdrant collection size for memory usage

## Deployment

### Docker

\`\`\`bash
docker build -t legal-search-api .
docker run -p 8000:8000 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  legal-search-api
\`\`\`

### Cloud Deployment

1. Set QDRANT_URL to your cloud Qdrant instance
2. Set OpenAI API key
3. Deploy with your preferred platform (Vercel, Railway, Render, etc.)

## Troubleshooting

### Qdrant Connection Error
- Ensure Qdrant is running and accessible
- Check QDRANT_URL in .env

### Embedding Generation Fails
- Verify OpenAI API key if using OpenAI embeddings
- Check internet connection for model downloads

### File Upload Fails
- Check file format is supported
- Verify file size < MAX_FILE_SIZE_MB
- Ensure upload directory exists and is writable

## License

MIT

## Support

For issues or questions, please refer to the documentation or create an issue.
