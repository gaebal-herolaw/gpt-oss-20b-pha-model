#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 검색 테스트 (LLM 없이 검색만 테스트)
"""

import sys
import io
from pathlib import Path

# Windows 인코딩 문제 해결
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store import VectorStore

def main():
    print("\n" + "="*60)
    print("Vector Search Test")
    print("="*60)

    # 벡터 스토어 초기화
    print("\nInitializing vector store...")
    vector_store = VectorStore()
    vector_store.create_collection(reset=False)

    # 통계 출력
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding model: {stats['embedding_model']}")

    # 테스트 쿼리
    test_queries = [
        "Personal Health Agent의 주요 기능은 무엇인가요?",
        "What are the key features of Personal Health Agent?",
        "건강 데이터 분석 방법"
    ]

    for query in test_queries:
        print("\n" + "="*60)
        print(f"Query: {query}")
        print("="*60)

        # 검색
        results = vector_store.search(query, k=3)

        # 결과 출력
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            relevance = round(1 - dist, 3)
            print(f"\n[Result {i}] Relevance: {relevance}")
            print(f"Source: {meta.get('source', 'Unknown')}")
            print(f"Content: {doc[:200]}...")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
