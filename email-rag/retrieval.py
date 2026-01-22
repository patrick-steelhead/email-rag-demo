"""
Email RAG Pipeline - Retrieval Module
Performs metadata-filtered vector search against pgvector for relevant email chunks.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

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

# Default retrieval settings
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))


@dataclass
class RetrievalFilters:
    """Filters for metadata-based retrieval."""
    email_type: Optional[str] = None
    sender_domain: Optional[str] = None
    domain_id: Optional[int] = None
    customer_id: Optional[int] = None
    vendor_id: Optional[int] = None
    date_from: Optional[str] = None  # ISO format
    date_to: Optional[str] = None    # ISO format


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its similarity score."""
    id: int
    message_id: str
    chunk_index: int
    content: str
    metadata: dict
    similarity_score: float


def get_db_connection():
    """Create a database connection."""
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    else:
        return psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            cursor_factory=RealDictCursor
        )


def get_embeddings_model() -> OpenAIEmbeddings:
    """Get the embeddings model."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )


def embed_query(query: str) -> list[float]:
    """Create an embedding for a query string."""
    embeddings_model = get_embeddings_model()
    return embeddings_model.embed_query(query)


def build_filter_clause(filters: Optional[RetrievalFilters]) -> tuple[str, list]:
    """
    Build SQL WHERE clause and parameters for metadata filtering.
    
    Args:
        filters: Optional filters to apply
    
    Returns:
        Tuple of (WHERE clause string, list of parameters)
    """
    if filters is None:
        return "", []
    
    conditions = []
    params = []
    
    if filters.email_type:
        conditions.append("metadata->>'email_type' = %s")
        params.append(filters.email_type)
    
    if filters.sender_domain:
        conditions.append("metadata->>'sender_domain' = %s")
        params.append(filters.sender_domain)
    
    if filters.domain_id is not None:
        conditions.append("(metadata->>'domain_id')::int = %s")
        params.append(filters.domain_id)
    
    if filters.customer_id is not None:
        conditions.append("(metadata->>'customer_id')::int = %s")
        params.append(filters.customer_id)
    
    if filters.vendor_id is not None:
        conditions.append("(metadata->>'vendor_id')::int = %s")
        params.append(filters.vendor_id)
    
    if filters.date_from:
        conditions.append("(metadata->>'created_at')::timestamp >= %s::timestamp")
        params.append(filters.date_from)
    
    if filters.date_to:
        conditions.append("(metadata->>'created_at')::timestamp <= %s::timestamp")
        params.append(filters.date_to)
    
    if conditions:
        return " AND " + " AND ".join(conditions), params
    else:
        return "", []


def retrieve_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: Optional[RetrievalFilters] = None
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant email chunks for a given query.
    
    Uses cosine similarity for vector search with optional metadata filtering.
    
    Args:
        query: The search query string
        top_k: Number of chunks to retrieve
        filters: Optional metadata filters
    
    Returns:
        List of RetrievedChunk objects sorted by similarity
    """
    # Create query embedding
    query_embedding = embed_query(query)
    
    # Build filter clause
    filter_clause, filter_params = build_filter_clause(filters)
    
    # Build the similarity search query
    # Using cosine distance: 1 - (a <=> b) gives cosine similarity
    search_query = f"""
    SELECT 
        id,
        message_id,
        chunk_index,
        content,
        metadata,
        1 - (embedding <=> %s::vector) as similarity_score
    FROM email_embeddings
    WHERE embedding IS NOT NULL
    {filter_clause}
    ORDER BY embedding <=> %s::vector
    LIMIT {top_k}
    """
    
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Convert embedding to string format for pgvector
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            # Parameter order: embedding (SELECT), filter params, embedding (ORDER BY)
            cur.execute(search_query, [embedding_str] + filter_params + [embedding_str])
            results = cur.fetchall()
        
        chunks = []
        for row in results:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            chunks.append(RetrievedChunk(
                id=row["id"],
                message_id=str(row["message_id"]),
                chunk_index=row["chunk_index"],
                content=row["content"],
                metadata=metadata,
                similarity_score=float(row["similarity_score"])
            ))
        
        return chunks
        
    finally:
        conn.close()


def retrieve_documents_with_scores(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: Optional[RetrievalFilters] = None
) -> list[tuple[RetrievedChunk, float]]:
    """
    Retrieve documents with their similarity scores.
    
    Returns:
        List of (chunk, score) tuples
    """
    chunks = retrieve_documents(query, top_k, filters)
    return [(chunk, chunk.similarity_score) for chunk in chunks]


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        chunks: List of retrieved chunks
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata
        sender = metadata.get("sender_address", "Unknown")
        subject = metadata.get("subject", "No subject")
        email_type = metadata.get("email_type", "Unknown")
        created_at = metadata.get("created_at", "Unknown date")
        
        # Format the date if present
        if created_at and created_at != "Unknown date":
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                created_at = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                pass
        
        context_parts.append(
            f"[Email {i}]\n"
            f"From: {sender}\n"
            f"Subject: {subject}\n"
            f"Type: {email_type}\n"
            f"Date: {created_at}\n"
            f"Content:\n{chunk.content}\n"
        )
    
    return "\n---\n".join(context_parts)


def debug_collection() -> dict:
    """Debug function to check the email embeddings collection status."""
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Total count
            cur.execute("SELECT COUNT(*) FROM email_embeddings")
            total_count = cur.fetchone()["count"]
            
            # Count by email type
            cur.execute("""
                SELECT metadata->>'email_type' as email_type, COUNT(*) as count
                FROM email_embeddings
                GROUP BY metadata->>'email_type'
                ORDER BY count DESC
                LIMIT 10
            """)
            email_types = cur.fetchall()
            
            # Count by domain
            cur.execute("""
                SELECT (metadata->>'domain_id')::int as domain_id, COUNT(*) as count
                FROM email_embeddings
                GROUP BY metadata->>'domain_id'
                ORDER BY count DESC
            """)
            domains = cur.fetchall()
            
            # Sample embedding dimensions
            cur.execute("""
                SELECT array_length(embedding::real[], 1) as dimensions
                FROM email_embeddings
                WHERE embedding IS NOT NULL
                LIMIT 1
            """)
            dimensions_result = cur.fetchone()
            dimensions = dimensions_result["dimensions"] if dimensions_result else None
        
        return {
            "total_chunks": total_count,
            "embedding_dimensions": dimensions,
            "email_types": [dict(r) for r in email_types],
            "domains": [dict(r) for r in domains],
        }
        
    finally:
        conn.close()


def main():
    """Test retrieval with a sample query."""
    print("=" * 60)
    print("Email RAG Pipeline - Retrieval Test")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Debug the collection
    print("\n📊 Collection Status:")
    stats = debug_collection()
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Embedding dimensions: {stats['embedding_dimensions']}")
    print(f"\n   Email types:")
    for et in stats["email_types"][:5]:
        print(f"     {et['email_type']}: {et['count']}")
    print(f"\n   Domains:")
    for d in stats["domains"]:
        print(f"     Domain {d['domain_id']}: {d['count']}")
    
    if stats["total_chunks"] == 0:
        print("\n❌ No documents found! Run ingestion.py first.")
        return
    
    # Test queries
    print("\n" + "=" * 60)
    print("Running Retrieval Tests")
    print("=" * 60)
    
    test_queries = [
        "What shipping confirmations have been sent?",
        "Are there any invoice-related emails?",
        "Show me emails about order status updates",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        chunks = retrieve_documents(query, top_k=3)
        
        if not chunks:
            print("   No results found")
            continue
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n   [{i}] Score: {chunk.similarity_score:.4f}")
            print(f"       Subject: {chunk.metadata.get('subject', 'N/A')[:50]}")
            print(f"       Type: {chunk.metadata.get('email_type', 'N/A')}")
            print(f"       Preview: {chunk.content[:100]}...")
    
    print("\n✅ Retrieval test complete")


if __name__ == "__main__":
    main()
