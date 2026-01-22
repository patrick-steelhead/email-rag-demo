# Email RAG Pipeline Demo

A Retrieval-Augmented Generation (RAG) pipeline for Steelhead email logs, implementing metadata-filtered search using pgvector.

## Overview

This demo implements a complete RAG pipeline that:
1. Extracts emails from Steelhead's `email_log` and `email_user_detail` tables (domains 1 and 70)
2. Chunks and embeds email content using OpenAI embeddings
3. Stores vectors in PostgreSQL using pgvector
4. Provides semantic search with metadata filtering
5. Generates contextual answers using GPT-4o-mini
6. Exposes a FastAPI service compatible with the Steelhead UI

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  dump_emails.py │ ──▶ │   ingestion.py   │ ──▶ │    pgvector     │
│  (Data Extract) │     │ (Chunk + Embed)  │     │  (Vector Store) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌──────────────────┐              │
│   generation.py │ ◀── │   retrieval.py   │ ◀────────────┘
│   (LLM Answer)  │     │ (Vector Search)  │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│     api.py      │ ◀── │   Steelhead UI   │
│  (FastAPI REST) │     │ (Sonar Chat)     │
└─────────────────┘     └──────────────────┘
```

## Why Metadata-Filtered RAG?

This implementation uses **Metadata-Filtered RAG** over other patterns (naive RAG, hybrid search, or graph RAG) for several reasons:

### 1. Rich Structured Metadata

Email data naturally contains structured metadata that users frequently want to filter on:
- **Email Type**: QUOTE, INVOICE, SHIPPING, CERTIFICATE, etc.
- **Domain ID**: Multi-tenant isolation for different Steelhead customers
- **Sender/Recipient**: Filter by specific people or companies
- **Date Range**: Time-based queries ("emails from last week")
- **Customer/Vendor ID**: Business relationship context

### 2. Improved Precision for Targeted Queries

When users ask specific questions like "show me invoice emails from last month," metadata filtering:
- Narrows the search space before vector similarity computation
- Eliminates irrelevant documents regardless of semantic similarity
- Produces higher precision results with fewer false positives

### 3. Lower Latency at Scale

For large email collections:
- Filtering reduces the number of vectors to compare
- PostgreSQL's GIN index on JSONB metadata enables fast filtering
- Combined with HNSW vector index for efficient similarity search

### 4. Simpler Architecture Than Graph RAG

While graph RAG excels at multi-hop reasoning across entities, email data is:
- Primarily document-centric (each email is self-contained)
- Already structured in relational tables
- Better served by metadata filtering than entity extraction

Graph RAG would add complexity (Neo4j, entity extraction, relationship modeling) without significant benefit for email search use cases.

### 5. Native PostgreSQL Integration

Using pgvector with JSONB metadata:
- No additional infrastructure (MongoDB, Pinecone, etc.)
- Transactional consistency with Steelhead's existing data
- Familiar SQL-based querying and debugging

## Quick Start

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- OpenAI API key
- Running Steelhead database with email data

### 2. Setup Environment

```bash
# Navigate to the email-rag directory
cd email-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp env.sample .env
# Edit .env with your credentials
```

### 3. Set Up pgvector

```bash
cd email-rag
python ingestion.py
```

Or execute manually:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS email_embeddings (
    id SERIAL PRIMARY KEY,
    message_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(message_id, chunk_index)
);

CREATE INDEX ON email_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON email_embeddings USING gin (metadata);
```

### 4. Extract Email Data

```bash
# Dump emails from domains 1 and 70
python dump_emails.py

# Or with a limit for testing
python dump_emails.py --limit 100
```

This creates `data/emails.json` with email documents and metadata.

### 5. Run Ingestion

```bash
# Ingest emails into pgvector
python ingestion.py
```

This will:
- Load email documents from JSON
- Chunk content (1000 chars, 200 overlap)
- Create OpenAI embeddings
- Store in pgvector with metadata

### 6. Test Retrieval

```bash
# Run retrieval tests
python retrieval.py
```

### 7. Test Generation

```bash
# Run interactive Q&A
python generation.py
```

### 8. Start the API Server

```bash
# Start FastAPI server
python api.py

# Or with uvicorn directly
uvicorn api:app --reload --port 8001
```

API endpoints:
- `GET /api/rag/health` - Health check with collection stats
- `POST /api/rag/chat` - Chat endpoint (Vercel AI SDK format)
- `POST /api/rag/chat/{channel_id}` - Sonar-compatible endpoint
- `GET /api/rag/search?query=...` - Direct search endpoint
- `GET /docs` - OpenAPI documentation

### 9. Connect Steelhead UI

> **Note:** The UI integration was not fully completed in the interest of time. This is planned to be revisited in a future iteration.

The UI has been modified to route Sonar chat to the RAG service. To use it:

1. Start the RAG API server (port 8001)
2. Configure your reverse proxy (nginx) to route `/api/rag/*` to the RAG service
3. Or update the UI to use the full URL (e.g., `http://localhost:8001/api/rag/chat/${channelId}`)

## Evaluation

The pipeline includes three evaluation modules to measure retrieval quality and performance:

### Precision Evaluation

Measures basic retrieval precision using LLM-as-judge:

```bash
# Run full evaluation
python evals/precision.py

# With custom k value
python evals/precision.py -k 10

# Save results to JSON
python evals/precision.py --output results.json

# Evaluate a single question
python evals/precision.py --question "What shipping emails have been sent?"
```

Metrics:
- **Precision@k**: Relevant documents / Total retrieved
- **Average Precision**: Mean precision across test queries

### Latency Evaluation

Compares retrieval and generation latency across different filter configurations:

```bash
# Run latency evaluation
python evals/latency.py

# With more runs for statistical accuracy
python evals/latency.py -n 5

# Skip generation latency (retrieval only)
python evals/latency.py --no-generation

# Save results to JSON
python evals/latency.py --output latency_results.json
```

Metrics:
- **Retrieval latency**: Time for vector search (ms)
- **Generation latency**: Time for LLM response (ms)
- **Total latency**: End-to-end time (ms)
- **Comparison**: Filtered vs unfiltered performance

Test cases include:
- No filter (baseline)
- Email type filters (INVOICE, SHIPPING, QUOTE)
- Domain ID filters
- Combined filters

### Precision Delta Evaluation

Measures the improvement in precision when using metadata filters vs. unfiltered search:

```bash
# Run precision delta evaluation
python evals/precision_delta.py

# With custom k value
python evals/precision_delta.py -k 10

# Include document details in output
python evals/precision_delta.py -v

# Save results to JSON
python evals/precision_delta.py --output delta_results.json
```

Metrics:
- **Unfiltered Precision**: Baseline precision without filters
- **Filtered Precision**: Precision with targeted metadata filters
- **Precision Delta**: Improvement from filtering (filtered - unfiltered)
- **By Filter Type**: Average delta grouped by filter category

This evaluation validates the benefit of metadata-filtered RAG by showing precision improvements for targeted queries.

## Project Structure

```
email-rag/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── env.sample              # Environment template
├── dump_emails.py          # Data extraction script
├── ingestion.py            # Chunking + embedding + storage
├── retrieval.py            # Vector search with filters
├── generation.py           # LLM answer generation
├── api.py                  # FastAPI REST service
├── data/                   # Extracted email data (gitignored)
│   └── emails.json
└── evals/
    ├── __init__.py         # Evaluation module exports
    ├── precision.py        # Retrieval precision evaluation
    ├── latency.py          # Latency comparison evaluation
    └── precision_delta.py  # Filtered vs unfiltered precision
```

## Configuration

Environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `POSTGRES_HOST` | localhost | Database host |
| `POSTGRES_PORT` | 10001 | Database port (exposed from Docker) |
| `POSTGRES_DB` | steelhead | Database name |
| `POSTGRES_USER` | steelhead | Database user |
| `POSTGRES_PASSWORD` | password | Database password |
| `CHUNK_SIZE` | 1000 | Text chunk size in characters |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 5 | Number of documents to retrieve |
| `LLM_MODEL` | gpt-4o-mini | OpenAI model for generation |
| `LLM_TEMPERATURE` | 0.0 | LLM temperature setting |
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8001 | API server port |

## Metadata Schema

Each email chunk is stored with the following metadata:

```json
{
    "message_id": "uuid",
    "sender_address": "user@example.com",
    "sender_name": "John Doe",
    "recipient_addresses": ["recipient@example.com"],
    "sender_domain": "example.com",
    "recipient_domains": ["example.com"],
    "email_type": "INVOICE",
    "source_message_id": "uuid or null",
    "created_at": "2024-01-15T10:30:00Z",
    "customer_id": 123,
    "vendor_id": null,
    "domain_id": 1,
    "subject": "Invoice #1234",
    "chunk_index": 0,
    "total_chunks": 3,
    "word_count": 150
}
```

## Reverting UI Changes

To revert the Sonar UI back to the original endpoint:

In `gosteelhead/packages/ui/src/Sonar/useSonarChannelInstance.tsx`, change:

```typescript
api: `/api/rag/chat/${channelId}`,
```

Back to:

```typescript
api: `/api/sonar/chat/${channelId}`,
```

## Troubleshooting

### "No documents found"
Run `python dump_emails.py` and `python ingestion.py` first.

### pgvector extension not found
Install pgvector: `CREATE EXTENSION vector;` (requires superuser)

### Connection refused on API
Ensure the API server is running: `python api.py`

### CORS errors
The API includes CORS middleware for development. For production, configure `allow_origins` appropriately.

## License

Internal use only - Steelhead Technologies.
