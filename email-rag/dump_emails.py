"""
Email RAG Pipeline - Email Data Dump Script
Extracts emails from steelhead.email_log and steelhead.email_user_detail tables
for domains 1 and 70, creating JSON files with metadata annotations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables - first try root .env file (same as other services)
root_env_path = Path(__file__).parent.parent / ".env"
if root_env_path.exists():
    load_dotenv(root_env_path)
# Then load local .env for overrides
load_dotenv()

# Database configuration - use same variable names as other services
# ROOT_USER_PASSWORD, EXPOSE_POSTGRES, etc. come from the root .env
ROOT_USER_PASSWORD = os.getenv("ROOT_USER_PASSWORD", "password")
EXPOSE_POSTGRES = os.getenv("EXPOSE_POSTGRES", "10001")

POSTGRES_HOST = os.getenv("HOST_POSTGRES", os.getenv("POSTGRES_HOST", "localhost"))
POSTGRES_PORT = os.getenv("PORT_POSTGRES", os.getenv("POSTGRES_PORT", EXPOSE_POSTGRES))
POSTGRES_DB = os.getenv("CONFIG_STORAGE", os.getenv("POSTGRES_DB", "steelhead"))
POSTGRES_USER = os.getenv("USER_POSTGRES", os.getenv("POSTGRES_USER", "steelhead"))
POSTGRES_PASSWORD = os.getenv("PASSWORD_POSTGRES", os.getenv("POSTGRES_PASSWORD", ROOT_USER_PASSWORD))

# Build DATABASE_URL if not explicitly set
DATABASE_URL = os.getenv("DATABASE_URL")

# Target domains for the demo
TARGET_DOMAINS = [1, 70]

# Output directory
OUTPUT_DIR = Path(__file__).parent / "data"


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


def extract_domain_from_email(email: str) -> Optional[str]:
    """Extract the domain part from an email address."""
    if not email or "@" not in email:
        return None
    return email.split("@")[-1].lower()


def fetch_emails(conn, domain_ids: list[int], limit: Optional[int] = None) -> list[dict]:
    """
    Fetch emails from email_log table with recipient details.
    
    Args:
        conn: Database connection
        domain_ids: List of domain IDs to filter by
        limit: Optional limit on number of emails to fetch
    
    Returns:
        List of email records with metadata
    """
    query = """
    SELECT 
        el.message_id,
        el.creator_email,
        el.creator_name,
        el.domain_id,
        el.created_at,
        el.email_type,
        el.subject,
        el.body,
        el.raw_body,
        el.source_message_id,
        el.customer_id,
        el.vendor_id,
        el.include_sender,
        el.visible_to_others,
        el.is_external_reply,
        COALESCE(
            (
                SELECT json_agg(json_build_object(
                    'email', eud.email,
                    'user_id', eud.user_id,
                    'read', eud.read,
                    'starred', eud.starred
                ))
                FROM steelhead.email_user_detail eud
                WHERE eud.message_id = el.message_id
            ),
            '[]'::json
        ) as recipients
    FROM steelhead.email_log el
    WHERE el.domain_id = ANY(%s)
        AND el.raw_body IS NOT NULL
        AND el.raw_body != ''
    ORDER BY el.created_at DESC
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    with conn.cursor() as cur:
        cur.execute(query, (domain_ids,))
        return cur.fetchall()


def transform_email_to_document(email: dict) -> dict:
    """
    Transform a raw email record into a document with metadata annotations.
    
    Args:
        email: Raw email record from database
    
    Returns:
        Document with content and metadata for RAG ingestion
    """
    # Extract sender domain
    sender_email = email.get("creator_email", "") or ""
    sender_domain = extract_domain_from_email(sender_email)
    
    # Extract recipient emails and domains
    recipients = email.get("recipients", []) or []
    if isinstance(recipients, str):
        recipients = json.loads(recipients)
    
    recipient_addresses = [r.get("email", "") for r in recipients if r.get("email")]
    recipient_domains = list(set(
        extract_domain_from_email(addr) 
        for addr in recipient_addresses 
        if extract_domain_from_email(addr)
    ))
    
    # Build the content string (subject + body for context)
    subject = email.get("subject", "") or ""
    raw_body = email.get("raw_body", "") or ""
    
    # Combine subject and body for the content
    content_parts = []
    if subject:
        content_parts.append(f"Subject: {subject}")
    if raw_body:
        content_parts.append(raw_body)
    
    content = "\n\n".join(content_parts)
    
    # Format created_at as ISO string
    created_at = email.get("created_at")
    if isinstance(created_at, datetime):
        created_at_str = created_at.isoformat()
    elif created_at:
        created_at_str = str(created_at)
    else:
        created_at_str = None
    
    # Format source_message_id
    source_message_id = email.get("source_message_id")
    if source_message_id:
        source_message_id = str(source_message_id)
    
    return {
        "message_id": str(email.get("message_id")),
        "sender_address": sender_email,
        "sender_name": email.get("creator_name"),
        "recipient_addresses": recipient_addresses,
        "sender_domain": sender_domain,
        "recipient_domains": recipient_domains,
        "email_type": email.get("email_type", "GENERIC"),
        "source_message_id": source_message_id,
        "created_at": created_at_str,
        "customer_id": email.get("customer_id"),
        "vendor_id": email.get("vendor_id"),
        "domain_id": email.get("domain_id"),
        "subject": subject,
        "content": content,
        "is_external_reply": email.get("is_external_reply", False),
        "visible_to_others": email.get("visible_to_others", True),
    }


def dump_emails(
    output_file: Optional[str] = None,
    limit: Optional[int] = None,
    pretty: bool = True
) -> list[dict]:
    """
    Main function to dump emails from the database.
    
    Args:
        output_file: Path to output JSON file (optional)
        limit: Maximum number of emails to dump (optional)
        pretty: Whether to pretty-print JSON output
    
    Returns:
        List of transformed email documents
    """
    print("=" * 60)
    print("Email RAG Pipeline - Data Dump")
    print("=" * 60)
    print(f"\nTarget domains: {TARGET_DOMAINS}")
    if limit:
        print(f"Limit: {limit} emails")
    
    # Connect to database
    if DATABASE_URL:
        print(f"\nConnecting to database via DATABASE_URL...")
    else:
        print(f"\nConnecting to database at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB} as {POSTGRES_USER}...")
    conn = get_db_connection()
    
    try:
        # Fetch emails
        print("Fetching emails...")
        raw_emails = fetch_emails(conn, TARGET_DOMAINS, limit)
        print(f"Found {len(raw_emails)} emails")
        
        # Transform to documents
        print("Transforming to documents...")
        documents = []
        for email in raw_emails:
            doc = transform_email_to_document(email)
            # Only include documents with actual content
            if doc["content"].strip():
                documents.append(doc)
        
        print(f"Created {len(documents)} documents with content")
        
        # Calculate statistics
        email_types = {}
        domains_count = {1: 0, 70: 0}
        for doc in documents:
            email_type = doc.get("email_type", "UNKNOWN")
            email_types[email_type] = email_types.get(email_type, 0) + 1
            domain_id = doc.get("domain_id")
            if domain_id in domains_count:
                domains_count[domain_id] += 1
        
        print(f"\nDocuments by domain:")
        for domain_id, count in domains_count.items():
            print(f"  Domain {domain_id}: {count}")
        
        print(f"\nDocuments by email type:")
        for email_type, count in sorted(email_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {email_type}: {count}")
        
        # Save to file
        if output_file:
            output_path = Path(output_file)
        else:
            OUTPUT_DIR.mkdir(exist_ok=True)
            output_path = OUTPUT_DIR / "emails.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(documents, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(documents, f, ensure_ascii=False, default=str)
        
        print(f"\nSaved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return documents
        
    finally:
        conn.close()


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dump Steelhead emails for RAG pipeline"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: data/emails.json)"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Maximum number of emails to dump"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no pretty printing)"
    )
    
    args = parser.parse_args()
    
    documents = dump_emails(
        output_file=args.output,
        limit=args.limit,
        pretty=not args.compact
    )
    
    print(f"\n✅ Successfully dumped {len(documents)} email documents")


if __name__ == "__main__":
    main()
