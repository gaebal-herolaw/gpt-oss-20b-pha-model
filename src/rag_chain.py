from typing import List, Dict, Any
from .local_llm import LocalGPT
from .vector_store import VectorStore

class RAGChain:
    """RAG 파이프라인"""
    
    def __init__(self, llm: LocalGPT = None, vector_store: VectorStore = None):
        self.llm = llm or LocalGPT()
        self.vector_store = vector_store or VectorStore()
        
        # 컬렉션 로드
        if self.vector_store.collection is None:
            self.vector_store.create_collection(reset=False)
        
        print("[OK] RAG Chain initialized")
    
    def create_prompt(self, query: str, contexts: List[str]) -> str:
        """프롬프트 생성"""
        
        # 컨텍스트 포맷팅
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            context_text += f"\n[문서 {i}]\n{ctx}\n"
        
        # 프롬프트 템플릿
        prompt = f"""당신은 Personal Health Agent 분야의 전문가입니다.
아래 논문 내용을 참고하여 사용자의 질문에 정확하고 상세하게 답변하세요.

참고 문서:
{context_text}

질문: {query}

답변 지침:
1. 논문 내용을 기반으로 핵심 정보를 명확하게 설명하세요
2. 구체적인 수치, 실험 결과, 예시를 포함하세요
3. 참고한 문서 번호를 명시하세요 (예: [문서 1], [문서 3])
4. 논문에 없는 내용은 추측하지 마세요

답변:"""
        
        return prompt
    
    def answer(
        self,
        query: str,
        k: int = 5,
        temperature: float = 0.7,
        max_length: int = 2048
    ) -> Dict[str, Any]:
        """질문에 답변"""
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # 1. 관련 문서 검색
        print("\n[1/3] Searching for relevant documents...")
        search_results = self.vector_store.search(query, k=k)
        
        contexts = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        distances = search_results['distances'][0]
        
        print(f"[OK] Found {len(contexts)} relevant chunks")
        
        # 2. 프롬프트 생성
        print("\n[2/3] Creating prompt...")
        prompt = self.create_prompt(query, contexts)
        print(f"[OK] Prompt created ({len(prompt)} characters)")
        
        # 3. LLM으로 답변 생성
        print("\n[3/3] Generating answer...")
        answer = self.llm.generate(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        print(f"[OK] Answer generated ({len(answer)} characters)")
        
        # 결과 반환
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "content": ctx[:200] + "...",
                    "metadata": meta,
                    "relevance_score": round(1 - dist, 3)
                }
                for ctx, meta, dist in zip(contexts, metadatas, distances)
            ],
            "num_sources": len(contexts)
        }
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """결과를 보기 좋게 포맷팅"""
        
        output = f"""
{'='*60}
질문: {result['query']}
{'='*60}

답변:
{result['answer']}

{'='*60}
참고 자료 ({result['num_sources']}개):
{'='*60}
"""
        
        for i, source in enumerate(result['sources'], 1):
            output += f"""
[{i}] (관련도: {source['relevance_score']})
출처: {source['metadata'].get('source', 'Unknown')}
내용: {source['content']}
"""
        
        return output

# 사용 예시
if __name__ == "__main__":
    # RAG 체인 초기화
    rag = RAGChain()
    
    # 질문
    query = "Personal Health Agent의 주요 기능과 특징을 설명해주세요."
    
    # 답변 생성
    result = rag.answer(query)
    
    # 출력
    print(rag.format_response(result))
