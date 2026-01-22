"""
Email RAG Pipeline - Generation Module
Uses retrieved email context to generate answers using OpenAI LLM.
"""

import os
from typing import Optional, Generator

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import (
    retrieve_documents,
    format_retrieved_context,
    RetrievalFilters,
    RetrievedChunk,
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
TOP_K = int(os.getenv("TOP_K", "5"))

# RAG Prompt Template for Email Q&A
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about company emails and communications.

Use ONLY the information from the email context below to answer the question. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what information is missing.

When referencing emails, include relevant details like:
- Who sent the email
- The subject line
- The date if available
- Key content that supports your answer

Email Context:
{context}

Question: {question}

Answer:"""

# Streaming prompt template
STREAMING_PROMPT_TEMPLATE = """You are a helpful AI assistant for Steelhead, a manufacturing ERP system. You help users find information about their business communications and emails.

Based on the email context provided below, answer the user's question. Be concise but thorough. If you reference specific emails, mention key details like sender, subject, and date.

If the context doesn't contain relevant information, let the user know and suggest what they might search for instead.

Email Context:
{context}

User Question: {question}

Provide a helpful, conversational response:"""


def create_rag_chain():
    """Create the RAG chain with prompt template and LLM."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    return chain


def create_streaming_chain():
    """Create a chain configured for streaming responses."""
    prompt = ChatPromptTemplate.from_template(STREAMING_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        streaming=True
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    filters: Optional[RetrievalFilters] = None,
    verbose: bool = False
) -> dict:
    """
    Generate an answer using the RAG pipeline.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        filters: Optional metadata filters for retrieval
        verbose: If True, include retrieved documents in response
    
    Returns:
        Dictionary containing the answer and sources
    """
    # Step 1: Retrieve relevant email chunks
    chunks = retrieve_documents(question, top_k=top_k, filters=filters)
    
    if not chunks:
        return {
            "answer": "I couldn't find any relevant emails to answer your question. "
                     "Try rephrasing your question or checking if the email data has been ingested.",
            "sources": [],
            "retrieved_count": 0
        }
    
    # Step 2: Format context
    context = format_retrieved_context(chunks)
    
    # Step 3: Generate answer
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    # Prepare response
    response = {
        "answer": answer,
        "sources": [
            {
                "message_id": chunk.message_id,
                "subject": chunk.metadata.get("subject", "Unknown"),
                "sender": chunk.metadata.get("sender_address", "Unknown"),
                "email_type": chunk.metadata.get("email_type", "Unknown"),
                "date": chunk.metadata.get("created_at", "Unknown"),
                "similarity_score": chunk.similarity_score
            }
            for chunk in chunks
        ],
        "retrieved_count": len(chunks)
    }
    
    if verbose:
        response["retrieved_documents"] = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "similarity_score": chunk.similarity_score
            }
            for chunk in chunks
        ]
    
    return response


def generate_answer_streaming(
    question: str,
    top_k: int = TOP_K,
    filters: Optional[RetrievalFilters] = None
) -> Generator[str, None, dict]:
    """
    Generate an answer using the RAG pipeline with streaming.
    
    Yields text chunks as they are generated.
    Returns final response dict with sources after completion.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        filters: Optional metadata filters for retrieval
    
    Yields:
        Text chunks from the LLM response
    
    Returns:
        Dictionary containing sources and metadata
    """
    # Step 1: Retrieve relevant email chunks
    chunks = retrieve_documents(question, top_k=top_k, filters=filters)
    
    if not chunks:
        yield "I couldn't find any relevant emails to answer your question. "
        yield "Try rephrasing your question or checking if the email data has been ingested."
        return {
            "sources": [],
            "retrieved_count": 0
        }
    
    # Step 2: Format context
    context = format_retrieved_context(chunks)
    
    # Step 3: Generate answer with streaming
    chain = create_streaming_chain()
    
    full_response = ""
    for chunk in chain.stream({
        "context": context,
        "question": question
    }):
        full_response += chunk
        yield chunk
    
    # Return metadata
    return {
        "sources": [
            {
                "message_id": chunk.message_id,
                "subject": chunk.metadata.get("subject", "Unknown"),
                "sender": chunk.metadata.get("sender_address", "Unknown"),
                "email_type": chunk.metadata.get("email_type", "Unknown"),
                "date": chunk.metadata.get("created_at", "Unknown"),
            }
            for chunk in chunks
        ],
        "retrieved_count": len(chunks),
        "full_response": full_response
    }


def interactive_mode():
    """Run an interactive Q&A session."""
    print("\n" + "=" * 60)
    print("Email RAG - Interactive Q&A")
    print("Ask questions about your company emails")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60)
    
    while True:
        print()
        question = input("Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        print("\n🔍 Retrieving relevant emails...")
        print("🤖 Generating answer...\n")
        
        try:
            result = generate_answer(question, verbose=False)
            
            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print(f"\n📧 Sources ({result['retrieved_count']} emails):")
            for source in result["sources"][:5]:
                score = source.get("similarity_score", 0)
                print(f"  • [{score:.2f}] {source['subject'][:50]}")
                print(f"    From: {source['sender']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Email RAG Pipeline - Generation")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    print(f"\nConfiguration:")
    print(f"  Model: {LLM_MODEL}")
    print(f"  Temperature: {LLM_TEMPERATURE}")
    print(f"  Top-K: {TOP_K}")
    
    # Example queries
    example_questions = [
        "What shipping confirmations have been sent recently?",
        "Are there any emails about invoice issues?",
        "Show me communications with customers about their orders",
    ]
    
    print("\n📋 Example Questions Available:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-" * 50)
    choice = input("Enter question number (1-3), 'i' for interactive mode, or your own question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        question = example_questions[int(choice) - 1]
        print(f"\n📝 Question: {question}\n")
        print("🔍 Retrieving relevant emails...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(question, verbose=True)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📧 Sources ({result['retrieved_count']} emails):")
        for source in result["sources"]:
            print(f"  • {source['subject'][:50]}")
            print(f"    From: {source['sender']}, Type: {source['email_type']}")
    elif choice:
        print(f"\n📝 Question: {choice}\n")
        print("🔍 Retrieving relevant emails...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(choice, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📧 Sources ({result['retrieved_count']} emails):")
        for source in result["sources"]:
            print(f"  • {source['subject'][:50]}")
            print(f"    From: {source['sender']}")
    else:
        print("\n👋 No question provided. Run again to try!")


if __name__ == "__main__":
    main()
