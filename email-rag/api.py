"""
Email RAG Pipeline - FastAPI REST Service
Provides streaming chat endpoint compatible with Vercel AI SDK / @ai-sdk/react.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import (
    retrieve_documents,
    format_retrieved_context,
    RetrievalFilters,
    debug_collection,
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
TOP_K = int(os.getenv("TOP_K", "5"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

# Initialize FastAPI app
app = FastAPI(
    title="Email RAG API",
    description="RAG pipeline for Steelhead email logs",
    version="1.0.0"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are Sonar, an AI assistant for Steelhead, a manufacturing ERP system. You help users find information about their business communications and emails.

Based on the email context provided below, answer the user's question. Be concise but thorough. If you reference specific emails, mention key details like sender, subject, and date.

If the context doesn't contain relevant information, let the user know and suggest what they might search for instead.

Email Context:
{context}

User Question: {question}

Provide a helpful, conversational response:"""


class ChatMessage(BaseModel):
    """A chat message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    messages: list[ChatMessage]
    channel_id: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    total_chunks: int
    embedding_dimensions: Optional[int]


def create_streaming_chain():
    """Create a LangChain chain for streaming responses."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        streaming=True
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def format_sse_message(data: dict) -> str:
    """Format data as Server-Sent Events message."""
    return f"data: {json.dumps(data)}\n\n"


async def generate_streaming_response(question: str, filters: Optional[RetrievalFilters] = None):
    """
    Generate streaming response compatible with Vercel AI SDK format.
    
    The format follows the AI SDK stream protocol:
    - Text chunks are sent as: 0:"text content"
    - Data annotations are sent as: 2:[{data}]
    - Finish signal is sent as: d:{"finishReason":"stop"}
    """
    # Step 1: Retrieve relevant email chunks
    chunks = retrieve_documents(question, top_k=TOP_K, filters=filters)
    
    if not chunks:
        # Send empty response message
        yield '0:"I couldn\'t find any relevant emails to answer your question. Try rephrasing or checking if email data has been ingested."\n'
        yield 'd:{"finishReason":"stop"}\n'
        return
    
    # Step 2: Format context
    context = format_retrieved_context(chunks)
    
    # Send sources as data annotation (format: 2:[data])
    sources_data = [
        {
            "message_id": chunk.message_id,
            "subject": chunk.metadata.get("subject", "Unknown"),
            "sender": chunk.metadata.get("sender_address", "Unknown"),
            "email_type": chunk.metadata.get("email_type", "Unknown"),
            "similarity_score": round(chunk.similarity_score, 4),
        }
        for chunk in chunks
    ]
    yield f'2:{json.dumps([{"sources": sources_data}])}\n'
    
    # Step 3: Generate answer with streaming
    chain = create_streaming_chain()
    
    async for chunk in chain.astream({
        "context": context,
        "question": question
    }):
        # Escape the text for the stream protocol
        escaped = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        yield f'0:"{escaped}"\n'
    
    # Send finish signal
    yield 'd:{"finishReason":"stop"}\n'


@app.get("/api/rag/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with collection stats."""
    try:
        stats = debug_collection()
        return HealthResponse(
            status="ok",
            total_chunks=stats["total_chunks"],
            embedding_dimensions=stats["embedding_dimensions"]
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {str(e)}",
            total_chunks=0,
            embedding_dimensions=None
        )


@app.post("/api/rag/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint compatible with Vercel AI SDK.
    
    Accepts messages array and returns streaming response.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Extract the last user message as the question
    last_user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Return streaming response
    return StreamingResponse(
        generate_streaming_response(last_user_message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/api/rag/chat/{channel_id}")
async def chat_with_channel(channel_id: int, request: Request):
    """
    Chat endpoint with channel ID for Sonar compatibility.
    
    This matches the Sonar API route pattern: /api/sonar/chat/{channelId}
    """
    # Parse the request body
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    # Extract messages from the body
    messages = body.get("messages", [])
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Extract the last user message
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            # Handle both content as string and content as parts
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from parts
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            last_user_message = content
            break
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Return streaming response
    return StreamingResponse(
        generate_streaming_response(last_user_message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/rag/search")
async def search_emails(
    query: str,
    top_k: int = 5,
    email_type: Optional[str] = None,
    domain_id: Optional[int] = None
):
    """
    Direct search endpoint for email retrieval without generation.
    
    Useful for debugging and testing retrieval quality.
    """
    filters = RetrievalFilters(
        email_type=email_type,
        domain_id=domain_id
    ) if email_type or domain_id else None
    
    chunks = retrieve_documents(query, top_k=top_k, filters=filters)
    
    return {
        "query": query,
        "filters": {
            "email_type": email_type,
            "domain_id": domain_id,
        },
        "results": [
            {
                "message_id": chunk.message_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "similarity_score": chunk.similarity_score,
            }
            for chunk in chunks
        ],
        "count": len(chunks)
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Email RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/rag/health",
            "chat": "POST /api/rag/chat",
            "chat_with_channel": "POST /api/rag/chat/{channel_id}",
            "search": "GET /api/rag/search?query=...",
        },
        "documentation": "/docs"
    }


def main():
    """Run the API server."""
    import uvicorn
    
    print("=" * 60)
    print("Email RAG API Server")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Host: {API_HOST}")
    print(f"  Port: {API_PORT}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Top-K: {TOP_K}")
    print(f"\nEndpoints:")
    print(f"  Health: http://{API_HOST}:{API_PORT}/api/rag/health")
    print(f"  Chat: POST http://{API_HOST}:{API_PORT}/api/rag/chat")
    print(f"  Docs: http://{API_HOST}:{API_PORT}/docs")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
