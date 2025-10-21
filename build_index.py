#!/usr/bin/env python3
"""
벡터 데이터베이스 구축 스크립트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_processor import DataProcessor
from src.vector_store import VectorStore

def main():
    """벡터 DB 구축"""
    
    print("\n" + "="*60)
    print("Building Vector Database")
    print("="*60)
    
    # GPU 정보
    Config.print_gpu_info()
    
    # 1. 데이터 처리
    print("\n[Step 1/3] Processing documents...")
    processor = DataProcessor()
    chunks = processor.process()
    
    # 2. 벡터 스토어 생성
    print("\n[Step 2/3] Creating vector store...")
    vector_store = VectorStore()
    vector_store.create_collection(reset=True)
    
    # 3. 문서 추가
    print("\n[Step 3/3] Adding documents...")
    vector_store.add_documents(chunks)
    
    # 통계
    stats = vector_store.get_stats()
    print("\n" + "="*60)
    print("Build Complete!")
    print("="*60)
    print(f"Collection: {stats['collection_name']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print("="*60)

if __name__ == "__main__":
    main()
