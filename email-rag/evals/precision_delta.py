"""
Email RAG Pipeline - Precision Delta Evaluation
Compares retrieval precision with and without metadata filters.

Measures the improvement in precision when using targeted filters
for email type-specific or domain-specific questions.

Precision Delta = Filtered Precision - Unfiltered Precision
"""

import os
import sys
import json
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
JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-Judge prompt for relevance assessment
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge for an email search system. Determine if the retrieved email content is relevant to answering the given question.

An email is RELEVANT if it contains information that would help answer the question.
An email is NOT RELEVANT if it contains no useful information for the question.

Consider the email's subject, sender, type, and content when making your decision.

Question: {question}

Retrieved Email:
Subject: {subject}
From: {sender}
Type: {email_type}
Content: {content}

Is this email relevant? Respond with ONLY "RELEVANT" or "NOT_RELEVANT"."""


# Test cases with questions that should benefit from filtering
# Each has a question and the filter that should improve precision
PRECISION_DELTA_TEST_CASES = [
    {
        "id": "shipping_type",
        "question": "What shipping confirmations have been sent to customers?",
        "filter_type": "email_type",
        "filters": {"email_type": "SHIPPING"},
        "description": "Shipping question with email type filter"
    },
    {
        "id": "invoice_type",
        "question": "What invoice emails have been sent?",
        "filter_type": "email_type",
        "filters": {"email_type": "INVOICE"},
        "description": "Invoice question with email type filter"
    },
    {
        "id": "quote_type",
        "question": "What quote or pricing emails have been sent?",
        "filter_type": "email_type",
        "filters": {"email_type": "QUOTE"},
        "description": "Quote question with email type filter"
    },
    {
        "id": "certificate_type",
        "question": "What certification or compliance emails have been sent?",
        "filter_type": "email_type",
        "filters": {"email_type": "CERTIFICATE"},
        "description": "Certificate question with email type filter"
    },
    {
        "id": "domain_70",
        "question": "What customer communications have happened?",
        "filter_type": "domain",
        "filters": {"domain_id": 70},
        "description": "Domain-specific question with domain filter"
    },
    {
        "id": "combined_shipping_domain",
        "question": "What shipping emails were sent in this domain?",
        "filter_type": "combined",
        "filters": {"email_type": "SHIPPING", "domain_id": 70},
        "description": "Combined email type + domain filter"
    },
    {
        "id": "combined_invoice_domain",
        "question": "What invoice emails were sent?",
        "filter_type": "combined",
        "filters": {"email_type": "INVOICE", "domain_id": 1},
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


def judge_relevance(judge_chain, question: str, chunk: RetrievedChunk) -> bool:
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


def calculate_precision(
    question: str, 
    chunks: list[RetrievedChunk], 
    judge
) -> tuple[float, list[bool]]:
    """
    Calculate precision for a set of retrieved chunks.
    
    Returns:
        Tuple of (precision_score, list_of_relevance_judgments)
    """
    if not chunks:
        return 0.0, []
    
    judgments = []
    for chunk in chunks:
        is_relevant = judge_relevance(judge, question, chunk)
        judgments.append(is_relevant)
    
    precision = sum(judgments) / len(judgments)
    return precision, judgments


def evaluate_precision_delta(
    question: str,
    filters: dict,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Compare precision with and without filters for a single question.
    
    Returns:
        Dictionary with precision comparison results
    """
    judge = create_relevance_judge()
    
    # Retrieve WITHOUT filters (baseline)
    unfiltered_chunks = retrieve_documents(question, top_k=k, filters=None)
    unfiltered_precision, unfiltered_judgments = calculate_precision(
        question, unfiltered_chunks, judge
    )
    
    # Retrieve WITH filters
    retrieval_filters = build_retrieval_filters(filters)
    filtered_chunks = retrieve_documents(question, top_k=k, filters=retrieval_filters)
    filtered_precision, filtered_judgments = calculate_precision(
        question, filtered_chunks, judge
    )
    
    # Calculate delta
    precision_delta = filtered_precision - unfiltered_precision
    
    result = {
        "question": question,
        "filters": filters,
        "k": k,
        "unfiltered": {
            "precision": unfiltered_precision,
            "relevant_count": sum(unfiltered_judgments),
            "total_docs": len(unfiltered_chunks),
            "judgments": unfiltered_judgments
        },
        "filtered": {
            "precision": filtered_precision,
            "relevant_count": sum(filtered_judgments),
            "total_docs": len(filtered_chunks),
            "judgments": filtered_judgments
        },
        "precision_delta": precision_delta,
        "improvement": precision_delta > 0
    }
    
    if verbose:
        result["unfiltered_subjects"] = [
            chunk.metadata.get("subject", "Unknown")[:50]
            for chunk in unfiltered_chunks
        ]
        result["filtered_subjects"] = [
            chunk.metadata.get("subject", "Unknown")[:50]
            for chunk in filtered_chunks
        ]
    
    return result


def run_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Run precision delta evaluation on all test cases.
    
    Returns:
        Dictionary with aggregate results
    """
    if test_cases is None:
        test_cases = PRECISION_DELTA_TEST_CASES
    
    print("=" * 60)
    print("Email RAG Pipeline - Precision Delta Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents): {k}")
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
    improvements = 0
    no_change = 0
    regressions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        description = test_case["description"]
        filter_type = test_case["filter_type"]
        
        print(f"\n[{i}/{len(test_cases)}] {test_id}")
        print(f"    Type: {filter_type}")
        print(f"    Q: {question[:55]}...")
        
        result = evaluate_precision_delta(
            question=question,
            filters=filters,
            k=k,
            verbose=verbose
        )
        result["test_id"] = test_id
        result["description"] = description
        result["filter_type"] = filter_type
        
        results.append(result)
        
        # Track outcomes
        delta = result["precision_delta"]
        if delta > 0:
            improvements += 1
            status = f"🟢 +{delta:.0%}"
        elif delta < 0:
            regressions += 1
            status = f"🔴 {delta:.0%}"
        else:
            no_change += 1
            status = "⚪ 0%"
        
        print(f"    Unfiltered: {result['unfiltered']['precision']:.0%} ({result['unfiltered']['relevant_count']}/{result['unfiltered']['total_docs']})")
        print(f"    Filtered:   {result['filtered']['precision']:.0%} ({result['filtered']['relevant_count']}/{result['filtered']['total_docs']})")
        print(f"    Delta:      {status}")
    
    # Calculate aggregate metrics
    avg_unfiltered = sum(r["unfiltered"]["precision"] for r in results) / len(results) if results else 0
    avg_filtered = sum(r["filtered"]["precision"] for r in results) / len(results) if results else 0
    avg_delta = sum(r["precision_delta"] for r in results) / len(results) if results else 0
    
    # Group by filter type
    by_filter_type = {}
    for r in results:
        ft = r["filter_type"]
        if ft not in by_filter_type:
            by_filter_type[ft] = []
        by_filter_type[ft].append(r["precision_delta"])
    
    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "avg_unfiltered_precision": avg_unfiltered,
        "avg_filtered_precision": avg_filtered,
        "avg_precision_delta": avg_delta,
        "improvements": improvements,
        "no_change": no_change,
        "regressions": regressions,
        "by_filter_type": {
            ft: sum(deltas) / len(deltas) 
            for ft, deltas in by_filter_type.items()
        },
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Average Unfiltered Precision: {avg_unfiltered:.2%}")
    print(f"   Average Filtered Precision:   {avg_filtered:.2%}")
    print(f"   Average Precision Delta:      {avg_delta:+.2%}")
    
    print(f"\n📈 Outcomes:")
    print(f"   🟢 Improvements: {improvements}/{len(results)} ({100*improvements/len(results):.0f}%)")
    print(f"   ⚪ No Change:    {no_change}/{len(results)} ({100*no_change/len(results):.0f}%)")
    print(f"   🔴 Regressions:  {regressions}/{len(results)} ({100*regressions/len(results):.0f}%)")
    
    print(f"\n📋 By Filter Type:")
    for ft, avg in summary["by_filter_type"].items():
        indicator = "🟢" if avg > 0 else "🔴" if avg < 0 else "⚪"
        print(f"   {indicator} {ft}: {avg:+.2%} avg delta")
    
    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        delta = r["precision_delta"]
        if delta > 0:
            indicator = "🟢"
        elif delta < 0:
            indicator = "🔴"
        else:
            indicator = "⚪"
        print(f"   {indicator} {r['test_id']}: {r['unfiltered']['precision']:.0%} → {r['filtered']['precision']:.0%} ({delta:+.0%})")
    
    # Interpretation
    print(f"\n📈 Interpretation:")
    if avg_delta > 0.1:
        print("   Metadata filtering significantly improves precision for targeted queries.")
    elif avg_delta > 0:
        print("   Metadata filtering provides modest precision improvements.")
    elif avg_delta == 0:
        print("   Metadata filtering has neutral impact on precision.")
    else:
        print("   Metadata filtering may reduce precision - review filter configurations.")
    
    return summary


def main():
    """Run the precision delta evaluation."""
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run precision delta evaluation on Email RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, 
                        help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Include document subjects in output")
    parser.add_argument("--output", type=str, 
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    summary = run_evaluation(k=args.k, verbose=args.verbose)
    
    if args.output and "error" not in summary:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
