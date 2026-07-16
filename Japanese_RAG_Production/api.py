"""Minimal FastAPI Backend for Japanese RAG System - DEBUG VERSION"""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import traceback

from src.retrieval import Retriever
from src.generation import Generator

load_dotenv()

app = FastAPI(
    title="Japanese RAG API",
    description="Production-ready RAG backend for Japanese documents",
    version="1.0.0"
)

retriever = Retriever()
generator = Generator()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    chunks_retrieved: int


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Main endpoint to ask questions to the RAG system."""
    try:
        retrieved_chunks = retriever.retrieve(request.question, top_k=request.top_k)

        if not retrieved_chunks:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        answer = generator.generate(request.question, retrieved_chunks)

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=retrieved_chunks,
            chunks_retrieved=len(retrieved_chunks)
        )

    except HTTPException:
        raise
    except Exception as e:
        # Print full error in terminal
        print("\n========== FULL ERROR ==========")
        traceback.print_exc()
        print("================================\n")
        
        # Also return the real error message temporarily
        raise HTTPException(
            status_code=500,
            detail=f"Real error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)