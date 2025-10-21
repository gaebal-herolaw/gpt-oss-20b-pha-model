#!/usr/bin/env python3
"""
gpt-oss-20b-PHA 메인 실행 파일
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.rag_chain import RAGChain

def main():
    """메인 함수"""
    
    print("\n" + "="*60)
    print("gpt-oss-20b-PHA - Personal Health Agent Q&A System")
    print("="*60)
    
    # GPU 정보 출력
    Config.print_gpu_info()
    
    # RAG 체인 초기화
    print("\nInitializing RAG Chain...")
    rag = RAGChain()
    
    print("\n" + "="*60)
    print("System ready! Type 'quit' to exit.")
    print("="*60 + "\n")
    
    # 대화형 루프
    while True:
        try:
            # 사용자 입력
            query = input("\n질문: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n프로그램을 종료합니다.")
                break
            
            # 답변 생성
            result = rag.answer(query)
            
            # 결과 출력
            print(rag.format_response(result))
            
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
