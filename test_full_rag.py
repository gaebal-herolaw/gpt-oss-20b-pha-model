#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 RAG 시스템 테스트 (벡터 검색 + LLM 답변 생성)
"""

import sys
import io
from pathlib import Path

# Windows 인코딩 문제 해결
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import RAGChain

def main():
    print("\n" + "="*60)
    print("Full RAG System Test")
    print("="*60)
    print("\nMake sure LM Studio server is running!")
    print("="*60 + "\n")

    # RAG 체인 초기화
    print("Initializing RAG Chain...")
    rag = RAGChain()

    # 테스트 질문들
    test_queries = [
        "Personal Health Agent의 주요 기능은 무엇인가요?",
        "What are the main capabilities of Personal Health Agent?",
    ]

    for i, query in enumerate(test_queries, 1):
        print("\n" + "="*60)
        print(f"Test Query {i}: {query}")
        print("="*60)

        try:
            # 답변 생성
            result = rag.answer(
                query=query,
                k=3,  # 상위 3개 문서 검색
                temperature=0.7,
                max_length=512  # 짧게 테스트
            )

            # 결과 출력
            print("\n" + "-"*60)
            print("ANSWER:")
            print("-"*60)
            print(result['answer'])

            print("\n" + "-"*60)
            print(f"SOURCES ({result['num_sources']}):")
            print("-"*60)
            for j, source in enumerate(result['sources'], 1):
                print(f"\n[{j}] Relevance: {source['relevance_score']}")
                print(f"    Source: {source['metadata'].get('source', 'Unknown')}")
                print(f"    Preview: {source['content'][:100]}...")

        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
