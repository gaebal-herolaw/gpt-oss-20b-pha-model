#!/usr/bin/env python3
"""
FastAPI 기반 REST API 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import RAGChain
from src.config import Config

# FastAPI 앱 초기화
app = FastAPI(
    title="gpt-oss-20b-PHA API",
    description="Personal Health Agent Q&A System",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 체인 (전역 변수)
rag_chain: Optional[RAGChain] = None

# 요청 모델
class QueryRequest(BaseModel):
    query: str
    k: int = 5
    temperature: float = 0.7
    max_length: int = 2048

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    num_sources: int

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global rag_chain
    
    print("\n" + "="*60)
    print("Starting gpt-oss-20b-PHA API Server")
    print("="*60)
    
    Config.print_gpu_info()
    
    print("\nInitializing RAG Chain...")
    rag_chain = RAGChain()
    
    print("\n✓ Server ready!")
    print(f"API running at http://{Config.API_HOST}:{Config.API_PORT}")
    print("="*60 + "\n")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "gpt-oss-20b-PHA API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    stats = rag_chain.vector_store.get_stats()
    
    return {
        "status": "healthy",
        "gpu_available": Config.DEVICE == "cuda",
        "vector_store": stats
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """질문 답변 엔드포인트"""
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # 답변 생성
        result = rag_chain.answer(
            query=request.query,
            k=request.k,
            temperature=request.temperature,
            max_length=request.max_length
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True  # 개발 모드
    )
