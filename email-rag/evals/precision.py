"""
Email RAG Pipeline - Precision Evaluation
Measures retrieval precision using LLM-as-judge to assess document relevance.

Precision = (Number of relevant documents retrieved) / (Total documents retrieved, k)
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve_documents, debug_collection

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation configuration
DEFAULT_K = 5
JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-Judge prompt for relevance assessment
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge for an email search system. Your task is to determine if the retrieved email content is relevant to answering the given question.

An email is RELEVANT if it contains information that would help answer the question, even if it doesn't fully answer it.
An email is NOT RELEVANT if it contains no useful information for answering the question.

Consider the email's subject, sender, and content when making your decision.

Question: {question}

Retrieved Email Content:
Subject: {subject}
From: {sender}
Type: {email_type}
Content: {content}

Is this email relevant to answering the question? 
Respond with ONLY "RELEVANT" or "NOT_RELEVANT" - nothing else."""


# Test cases for email-specific evaluation
TEST_CASES = [
    {
        "id": "shipping_emails",
        "question": "What shipping confirmations have been sent?",
        "description": "Finding shipping-related communications"
    },
    {
        "id": "invoice_emails",
        "question": "Are there any emails about invoices or billing?",
        "description": "Financial document communications"
    },
    {
        "id": "order_status",
        "question": "What order status updates have been communicated?",
        "description": "Order tracking communications"
    },
    {
        "id": "customer_communications",
        "question": "What emails have been sent to customers?",
        "description": "Customer-facing communications"
    },
    {
        "id": "quote_emails",
        "question": "Are there any quote or pricing related emails?",
        "description": "Sales quotation communications"
    },
    {
        "id": "certificate_emails",
        "question": "Have any certification or compliance emails been sent?",
        "description": "Certification document communications"
    },
    {
        "id": "vendor_communications",
        "question": "What communications have there been with vendors?",
        "description": "Vendor-facing communications"
    },
    {
        "id": "recent_emails",
        "question": "What are the most recent email communications?",
        "description": "Recent activity check"
    },
]


def create_relevance_judge():
    """Create the LLM judge for assessing relevance."""
    prompt = ChatPromptTemplate.from_template(RELEVANCE_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def judge_relevance(judge_chain, question: str, chunk) -> bool:
    """Use LLM to judge if an email chunk is relevant to the question."""
    metadata = chunk.metadata
    
    response = judge_chain.invoke({
        "question": question,
        "subject": metadata.get("subject", "No subject"),
        "sender": metadata.get("sender_address", "Unknown"),
        "email_type": metadata.get("email_type", "Unknown"),
        "content": chunk.content[:1500]  # Limit content length
    })
    
    return "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()


def calculate_precision(question: str, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Calculate precision for a single query.
    
    Precision = relevant_documents / k
    
    Returns dict with precision score and details.
    """
    # Retrieve documents
    chunks = retrieve_documents(question, top_k=k)
    
    if not chunks:
        return {
            "question": question,
            "k": k,
            "retrieved": 0,
            "relevant": 0,
            "precision": 0.0,
            "judgments": []
        }
    
    # Judge each document
    judge = create_relevance_judge()
    judgments = []
    relevant_count = 0
    
    for i, chunk in enumerate(chunks):
        is_relevant = judge_relevance(judge, question, chunk)
        
        if is_relevant:
            relevant_count += 1
        
        judgment = {
            "doc_index": i + 1,
            "relevant": is_relevant,
            "subject": chunk.metadata.get("subject", "Unknown")[:50],
            "email_type": chunk.metadata.get("email_type", "Unknown"),
            "similarity_score": round(chunk.similarity_score, 4),
        }
        
        if verbose:
            judgment["content_preview"] = chunk.content[:200] + "..."
        
        judgments.append(judgment)
    
    precision = relevant_count / len(chunks)
    
    return {
        "question": question,
        "k": k,
        "retrieved": len(chunks),
        "relevant": relevant_count,
        "precision": precision,
        "judgments": judgments
    }


def run_evaluation(test_cases: list = None, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Run precision evaluation on all test cases.
    
    Returns aggregate results and per-question breakdown.
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    print("=" * 60)
    print("Email RAG Pipeline - Precision Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents retrieved): {k}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    
    # Check collection status
    stats = debug_collection()
    print(f"\nCollection Status:")
    print(f"  - Total chunks: {stats['total_chunks']}")
    
    if stats['total_chunks'] == 0:
        print("\n❌ No documents found! Run ingestion.py first.")
        return {"error": "No documents in collection"}
    
    print("\n" + "-" * 60)
    
    results = []
    total_relevant = 0
    total_retrieved = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {question[:60]}...")
        
        result = calculate_precision(question, k=k, verbose=verbose)
        result["test_id"] = test_case["id"]
        result["description"] = test_case.get("description", "")
        
        results.append(result)
        total_relevant += result["relevant"]
        total_retrieved += result["retrieved"]
        
        print(f"    Precision: {result['precision']:.2%} ({result['relevant']}/{result['retrieved']} relevant)")
        
        # Show individual judgments
        for j in result["judgments"]:
            status = "✅" if j["relevant"] else "❌"
            print(f"      {status} [{j['similarity_score']:.3f}] {j['email_type']}: {j['subject']}")
    
    # Calculate aggregate metrics
    avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
    overall_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    
    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "total_documents_retrieved": total_retrieved,
        "total_relevant_documents": total_relevant,
        "average_precision": avg_precision,
        "overall_precision": overall_precision,
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Average Precision (per query):  {avg_precision:.2%}")
    print(f"   Overall Precision (all docs):   {overall_precision:.2%}")
    print(f"   Total Relevant / Total Retrieved: {total_relevant} / {total_retrieved}")
    
    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        status = "✅" if r["precision"] >= 0.6 else "⚠️" if r["precision"] >= 0.4 else "❌"
        print(f"   {status} {r['test_id']}: {r['precision']:.2%}")
    
    # Interpretation
    print(f"\n📈 Interpretation:")
    if avg_precision >= 0.8:
        print("   Excellent retrieval quality! Most retrieved documents are relevant.")
    elif avg_precision >= 0.6:
        print("   Good retrieval quality. Consider tuning for better precision.")
    elif avg_precision >= 0.4:
        print("   Moderate retrieval quality. Review chunking and embedding strategies.")
    else:
        print("   Low retrieval quality. Consider improving data quality or retrieval approach.")
    
    return summary


def main():
    """Run the precision evaluation."""
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run precision evaluation on Email RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, 
                        help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Include document previews in output")
    parser.add_argument("--output", type=str, 
                        help="Save results to JSON file")
    parser.add_argument("--question", type=str, 
                        help="Evaluate a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        # Single question evaluation
        print(f"\nEvaluating single question with k={args.k}")
        result = calculate_precision(args.question, k=args.k, verbose=True)
        
        print(f"\nQuestion: {result['question']}")
        print(f"Precision: {result['precision']:.2%} ({result['relevant']}/{result['retrieved']} relevant)")
        print("\nJudgments:")
        for j in result["judgments"]:
            status = "✅" if j["relevant"] else "❌"
            print(f"  {status} [{j['similarity_score']:.3f}] {j['email_type']}: {j['subject']}")
            if args.verbose and "content_preview" in j:
                print(f"      {j['content_preview']}")
    else:
        # Full evaluation
        summary = run_evaluation(k=args.k, verbose=args.verbose)
        
        if args.output and "error" not in summary:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
