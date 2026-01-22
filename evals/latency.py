"""
Email RAG Pipeline - Latency Evaluation
Compares retrieval and generation latency with different metadata filter configurations.

Measures:
- Retrieval latency (vector search time)
- Generation latency (LLM response time)
- Total end-to-end latency
- Comparison: filtered vs unfiltered
"""

import os
import sys
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve_documents, RetrievalFilters, RetrievedChunk, debug_collection

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation configuration
DEFAULT_K = 5
DEFAULT_RUNS = 3  # Number of runs per test for averaging
GENERATION_MODEL = "gpt-4o-mini"

# RAG Prompt Template
RAG_PROMPT = """Answer based on the email context below.

Context:
{context}

Question: {question}

Answer:"""


def get_date_range_last_year() -> tuple[str, str]:
    """Get date range for the last year in ISO format."""
    now = datetime.now()
    date_to = now.isoformat()
    date_from = (now - timedelta(days=365)).isoformat()
    return date_from, date_to


def get_date_range_last_month() -> tuple[str, str]:
    """Get date range for the last month in ISO format."""
    now = datetime.now()
    date_to = now.isoformat()
    date_from = (now - timedelta(days=30)).isoformat()
    return date_from, date_to


# Test cases with different filter configurations
LATENCY_TEST_CASES = [
    {
        "id": "no_filter",
        "question": "What emails have been sent recently?",
        "filters": None,
        "description": "No filters (baseline)"
    },
    {
        "id": "email_type_invoice",
        "question": "What invoice-related emails have been sent?",
        "filters": {"email_type": "INVOICE"},
        "description": "Email type filter (INVOICE)"
    },
    {
        "id": "email_type_shipping",
        "question": "What shipping confirmations have been sent?",
        "filters": {"email_type": "SHIPPING"},
        "description": "Email type filter (SHIPPING)"
    },
    {
        "id": "email_type_quote",
        "question": "What quote emails have been sent?",
        "filters": {"email_type": "QUOTE"},
        "description": "Email type filter (QUOTE)"
    },
    {
        "id": "domain_filter_1",
        "question": "What communications have occurred?",
        "filters": {"domain_id": 1},
        "description": "Domain ID filter (domain 1)"
    },
    {
        "id": "domain_filter_70",
        "question": "What communications have occurred?",
        "filters": {"domain_id": 70},
        "description": "Domain ID filter (domain 70)"
    },
    {
        "id": "combined_type_domain",
        "question": "What shipping emails have been sent?",
        "filters": {"email_type": "SHIPPING", "domain_id": 70},
        "description": "Combined email type + domain filter"
    },
]


def build_retrieval_filters(filter_dict: Optional[dict]) -> Optional[RetrievalFilters]:
    """Convert a filter dictionary to a RetrievalFilters object."""
    if filter_dict is None:
        return None
    
    return RetrievalFilters(
        email_type=filter_dict.get("email_type"),
        sender_domain=filter_dict.get("sender_domain"),
        domain_id=filter_dict.get("domain_id"),
        customer_id=filter_dict.get("customer_id"),
        vendor_id=filter_dict.get("vendor_id"),
        date_from=filter_dict.get("date_from"),
        date_to=filter_dict.get("date_to"),
    )


def measure_retrieval_latency(
    query: str,
    k: int = DEFAULT_K,
    filters: Optional[dict] = None
) -> tuple[list[RetrievedChunk], float]:
    """
    Measure retrieval latency and return documents.
    
    Returns:
        Tuple of (chunks, latency_ms)
    """
    retrieval_filters = build_retrieval_filters(filters)
    
    start_time = time.perf_counter()
    
    chunks = retrieve_documents(
        query=query,
        top_k=k,
        filters=retrieval_filters
    )
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    return chunks, latency_ms


def measure_generation_latency(question: str, context: str) -> tuple[str, float]:
    """
    Measure generation latency and return answer.
    
    Returns:
        Tuple of (answer, latency_ms)
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    chain = prompt | llm | StrOutputParser()
    
    start_time = time.perf_counter()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    return answer, latency_ms


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format chunks into context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        subject = chunk.metadata.get("subject", "Unknown")[:100]
        sender = chunk.metadata.get("sender_address", "Unknown")
        parts.append(f"[{i}] From: {sender}\nSubject: {subject}\n{chunk.content[:500]}")
    return "\n\n".join(parts)


def run_single_test(
    question: str,
    filters: Optional[dict],
    k: int = DEFAULT_K,
    include_generation: bool = True
) -> dict:
    """
    Run a single latency test.
    
    Returns:
        Dictionary with latency measurements
    """
    # Measure retrieval
    chunks, retrieval_latency = measure_retrieval_latency(
        query=question,
        k=k,
        filters=filters
    )
    
    result = {
        "retrieval_latency_ms": retrieval_latency,
        "documents_retrieved": len(chunks),
    }
    
    # Measure generation if requested
    if include_generation and chunks:
        context = format_context(chunks)
        answer, generation_latency = measure_generation_latency(question, context)
        
        result["generation_latency_ms"] = generation_latency
        result["total_latency_ms"] = retrieval_latency + generation_latency
    else:
        result["generation_latency_ms"] = 0
        result["total_latency_ms"] = retrieval_latency
    
    return result


def run_latency_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    num_runs: int = DEFAULT_RUNS,
    include_generation: bool = True
) -> dict:
    """
    Run latency evaluation across all test cases.
    
    Args:
        test_cases: List of test cases to run
        k: Number of documents to retrieve
        num_runs: Number of runs per test for averaging
        include_generation: Whether to measure generation latency
    
    Returns:
        Dictionary with evaluation results
    """
    if test_cases is None:
        test_cases = LATENCY_TEST_CASES
    
    print("=" * 60)
    print("Email RAG Pipeline - Latency Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents): {k}")
    print(f"  - Runs per test: {num_runs}")
    print(f"  - Include generation: {include_generation}")
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
    
    for test_case in test_cases:
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        description = test_case["description"]
        
        print(f"\n📝 {test_id}: {description}")
        
        # Run multiple times and collect measurements
        retrieval_times = []
        generation_times = []
        total_times = []
        docs_retrieved = []
        
        for run in range(num_runs):
            run_result = run_single_test(
                question=question,
                filters=filters,
                k=k,
                include_generation=include_generation
            )
            
            retrieval_times.append(run_result["retrieval_latency_ms"])
            generation_times.append(run_result["generation_latency_ms"])
            total_times.append(run_result["total_latency_ms"])
            docs_retrieved.append(run_result["documents_retrieved"])
        
        # Calculate statistics
        result = {
            "test_id": test_id,
            "description": description,
            "filters": filters,
            "num_runs": num_runs,
            "documents_retrieved": docs_retrieved[0],
            "retrieval": {
                "mean_ms": statistics.mean(retrieval_times),
                "median_ms": statistics.median(retrieval_times),
                "min_ms": min(retrieval_times),
                "max_ms": max(retrieval_times),
                "stdev_ms": statistics.stdev(retrieval_times) if num_runs > 1 else 0
            },
            "generation": {
                "mean_ms": statistics.mean(generation_times),
                "median_ms": statistics.median(generation_times),
                "min_ms": min(generation_times),
                "max_ms": max(generation_times),
            },
            "total": {
                "mean_ms": statistics.mean(total_times),
                "median_ms": statistics.median(total_times),
            }
        }
        
        results.append(result)
        
        # Print summary for this test
        print(f"   Retrieval: {result['retrieval']['mean_ms']:.1f}ms (±{result['retrieval']['stdev_ms']:.1f}ms)")
        if include_generation:
            print(f"   Generation: {result['generation']['mean_ms']:.1f}ms")
            print(f"   Total: {result['total']['mean_ms']:.1f}ms")
        print(f"   Docs retrieved: {result['documents_retrieved']}")
    
    # Calculate summary statistics
    baseline = next((r for r in results if r["test_id"] == "no_filter"), results[0])
    baseline_retrieval = baseline["retrieval"]["mean_ms"]
    
    summary = {
        "k": k,
        "num_runs": num_runs,
        "baseline_retrieval_ms": baseline_retrieval,
        "results": results
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)
    print(f"\n{'Test Case':<25} {'Retrieval':<15} {'vs Baseline':<20} {'Docs':<10}")
    print("-" * 70)
    
    for r in results:
        retrieval = r["retrieval"]["mean_ms"]
        diff = retrieval - baseline_retrieval
        diff_pct = (diff / baseline_retrieval * 100) if baseline_retrieval > 0 else 0
        
        if diff < -5:  # More than 5ms faster
            diff_str = f"🟢 {diff:+.1f}ms ({diff_pct:+.0f}%)"
        elif diff > 5:  # More than 5ms slower
            diff_str = f"🔴 {diff:+.1f}ms ({diff_pct:+.0f}%)"
        else:
            diff_str = f"⚪ {diff:+.1f}ms (baseline)" if r["test_id"] == "no_filter" else f"⚪ {diff:+.1f}ms"
        
        print(f"{r['test_id']:<25} {retrieval:<15.1f} {diff_str:<20} {r['documents_retrieved']:<10}")
    
    if include_generation:
        print(f"\n📊 End-to-End Latency Summary:")
        for r in results:
            print(f"   {r['test_id']}: {r['total']['mean_ms']:.0f}ms total")
    
    # Interpretation
    print(f"\n📈 Interpretation:")
    faster_count = sum(1 for r in results if r["retrieval"]["mean_ms"] < baseline_retrieval - 5)
    slower_count = sum(1 for r in results if r["retrieval"]["mean_ms"] > baseline_retrieval + 5)
    
    if faster_count > 0:
        print(f"   {faster_count} filter configurations were faster than baseline")
    if slower_count > 0:
        print(f"   {slower_count} filter configurations were slower than baseline")
    
    avg_retrieval = statistics.mean(r["retrieval"]["mean_ms"] for r in results)
    print(f"   Average retrieval latency across all tests: {avg_retrieval:.1f}ms")
    
    return summary


def main():
    """Run the latency evaluation."""
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run latency evaluation on Email RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, 
                        help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-n", "--num-runs", type=int, default=DEFAULT_RUNS, 
                        help=f"Number of runs per test (default: {DEFAULT_RUNS})")
    parser.add_argument("--no-generation", action="store_true", 
                        help="Skip generation latency measurement")
    parser.add_argument("--output", type=str, 
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    summary = run_latency_evaluation(
        k=args.k,
        num_runs=args.num_runs,
        include_generation=not args.no_generation
    )
    
    if args.output and "error" not in summary:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
