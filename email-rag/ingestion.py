"""
Email RAG Pipeline - Ingestion Module
Loads email documents, chunks them, creates embeddings, and stores in pgvector.
"""

import json
import os
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables - first try root .env file (same as other services)
root_env_path = Path(__file__).parent.parent / ".env"
if root_env_path.exists():
    load_dotenv(root_env_path)
# Then load local .env for overrides
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database configuration - use same variable names as other services
ROOT_USER_PASSWORD = os.getenv("ROOT_USER_PASSWORD", "password")
EXPOSE_POSTGRES = os.getenv("EXPOSE_POSTGRES", "10001")

POSTGRES_HOST = os.getenv("HOST_POSTGRES", os.getenv("POSTGRES_HOST", "localhost"))
POSTGRES_PORT = os.getenv("PORT_POSTGRES", os.getenv("POSTGRES_PORT", EXPOSE_POSTGRES))
POSTGRES_DB = os.getenv("CONFIG_STORAGE", os.getenv("POSTGRES_DB", "steelhead"))
POSTGRES_USER = os.getenv("USER_POSTGRES", os.getenv("POSTGRES_USER", "steelhead"))
POSTGRES_PASSWORD = os.getenv("PASSWORD_POSTGRES", os.getenv("POSTGRES_PASSWORD", ROOT_USER_PASSWORD))
DATABASE_URL = os.getenv("DATABASE_URL")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Default data path
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_INPUT_FILE = DATA_DIR / "emails.json"


def get_db_connection():
    """Create a database connection."""
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    else:
        return psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )


def load_email_documents(input_file: Path) -> list[dict]:
    """Load email documents from JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk email documents using RecursiveCharacterTextSplitter.
    
    Each chunk includes the original metadata plus chunk-specific info.
    
    Args:
        documents: List of email documents with content and metadata
    
    Returns:
        List of chunks with content and metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    
    for doc in documents:
        content = doc.get("content", "")
        if not content.strip():
            continue
        
        # Split the content
        texts = text_splitter.split_text(content)
        
        # Create chunks with metadata
        for chunk_idx, text in enumerate(texts):
            chunk = {
                "message_id": doc.get("message_id"),
                "chunk_index": chunk_idx,
                "content": text,
                "metadata": {
                    "message_id": doc.get("message_id"),
                    "sender_address": doc.get("sender_address"),
                    "sender_name": doc.get("sender_name"),
                    "recipient_addresses": doc.get("recipient_addresses", []),
                    "sender_domain": doc.get("sender_domain"),
                    "recipient_domains": doc.get("recipient_domains", []),
                    "email_type": doc.get("email_type"),
                    "source_message_id": doc.get("source_message_id"),
                    "created_at": doc.get("created_at"),
                    "customer_id": doc.get("customer_id"),
                    "vendor_id": doc.get("vendor_id"),
                    "domain_id": doc.get("domain_id"),
                    "subject": doc.get("subject"),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(texts),
                    "word_count": len(text.split()),
                }
            }
            all_chunks.append(chunk)
    
    return all_chunks


def create_embeddings(chunks: list[dict], batch_size: int = 100) -> list[dict]:
    """
    Create embeddings for all chunks using OpenAI.
    
    Args:
        chunks: List of chunks with content
        batch_size: Number of texts to embed at once
    
    Returns:
        Chunks with embedding field added
    """
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["content"] for chunk in batch]
        
        print(f"  Embedding batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}...")
        embeddings = embeddings_model.embed_documents(texts)
        
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
    
    return chunks


def setup_database(conn) -> None:
    """Ensure the email_embeddings table exists."""
    setup_sql = """
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
    
    CREATE INDEX IF NOT EXISTS email_embeddings_embedding_idx 
    ON email_embeddings 
    USING hnsw (embedding vector_cosine_ops);
    
    CREATE INDEX IF NOT EXISTS email_embeddings_metadata_idx 
    ON email_embeddings 
    USING gin (metadata);
    
    CREATE INDEX IF NOT EXISTS email_embeddings_message_id_idx 
    ON email_embeddings (message_id);
    """
    
    with conn.cursor() as cur:
        cur.execute(setup_sql)
    conn.commit()


def clear_existing_embeddings(conn) -> int:
    """Clear existing embeddings from the table."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM email_embeddings")
        deleted = cur.rowcount
    conn.commit()
    return deleted


def store_embeddings(conn, chunks: list[dict], batch_size: int = 100) -> int:
    """
    Store chunks with embeddings in pgvector.
    
    Args:
        conn: Database connection
        chunks: List of chunks with embeddings
        batch_size: Number of rows to insert at once
    
    Returns:
        Number of rows inserted
    """
    insert_sql = """
    INSERT INTO email_embeddings (message_id, chunk_index, content, embedding, metadata)
    VALUES %s
    ON CONFLICT (message_id, chunk_index) 
    DO UPDATE SET 
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding,
        metadata = EXCLUDED.metadata,
        created_at = NOW()
    """
    
    total_inserted = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        values = [
            (
                chunk["message_id"],
                chunk["chunk_index"],
                chunk["content"],
                chunk["embedding"],
                json.dumps(chunk["metadata"])
            )
            for chunk in batch
        ]
        
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, values)
            total_inserted += len(batch)
        
        conn.commit()
        print(f"  Stored batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}...")
    
    return total_inserted


def run_ingestion(
    input_file: Optional[Path] = None,
    clear_existing: bool = True
) -> dict:
    """
    Run the full ingestion pipeline.
    
    Args:
        input_file: Path to input JSON file
        clear_existing: Whether to clear existing embeddings first
    
    Returns:
        Statistics about the ingestion
    """
    print("=" * 60)
    print("Email RAG Pipeline - Ingestion")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Determine input file
    if input_file is None:
        input_file = DEFAULT_INPUT_FILE
    
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Run dump_emails.py first to create the data file."
        )
    
    print(f"\nInput file: {input_file}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
    # Step 1: Load documents
    print("\n[1/5] Loading email documents...")
    documents = load_email_documents(input_file)
    print(f"  Loaded {len(documents)} documents")
    
    # Step 2: Chunk documents
    print("\n[2/5] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")
    
    # Calculate average chunks per document
    if documents:
        avg_chunks = len(chunks) / len(documents)
        print(f"  Average chunks per document: {avg_chunks:.1f}")
    
    # Step 3: Connect to database and setup
    print("\n[3/5] Setting up database...")
    conn = get_db_connection()
    
    try:
        setup_database(conn)
        print("  Database schema ready")
        
        if clear_existing:
            deleted = clear_existing_embeddings(conn)
            if deleted > 0:
                print(f"  Cleared {deleted} existing embeddings")
        
        # Step 4: Create embeddings
        print("\n[4/5] Creating embeddings...")
        chunks = create_embeddings(chunks)
        print(f"  Created embeddings for {len(chunks)} chunks")
        
        # Step 5: Store in database
        print("\n[5/5] Storing in pgvector...")
        stored = store_embeddings(conn, chunks)
        print(f"  Stored {stored} chunks")
        
        # Verify storage
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM email_embeddings")
            total_in_db = cur.fetchone()[0]
        
        stats = {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_stored": stored,
            "total_in_database": total_in_db,
        }
        
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"\n✅ Statistics:")
        print(f"   Documents loaded: {stats['documents_loaded']}")
        print(f"   Chunks created: {stats['chunks_created']}")
        print(f"   Chunks stored: {stats['chunks_stored']}")
        print(f"   Total in database: {stats['total_in_database']}")
        
        return stats
        
    finally:
        conn.close()


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest email documents into pgvector"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input JSON file path (default: data/emails.json)"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing embeddings before ingestion"
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input) if args.input else None
    
    run_ingestion(
        input_file=input_file,
        clear_existing=not args.no_clear
    )


if __name__ == "__main__":
    main()
