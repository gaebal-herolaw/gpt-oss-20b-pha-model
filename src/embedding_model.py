import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
from .config import Config

class LocalEmbedding:
    """GPU 기반 로컬 임베딩 모델"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = Config.DEVICE
        
        print(f"Loading embedding model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # 모델 로드
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # 메모리 확인
        if torch.cuda.is_available():
            print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 임베딩으로 변환"""
        print(f"Embedding {len(texts)} documents...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            device=self.device,
            normalize_embeddings=True  # 코사인 유사도 최적화
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩으로 변환"""
        embedding = self.model.encode(
            text,
            device=self.device,
            normalize_embeddings=True
        )
        
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        """임베딩 차원"""
        return self.model.get_sentence_embedding_dimension()

# 사용 예시
if __name__ == "__main__":
    embedder = LocalEmbedding()
    print(f"Embedding dimension: {embedder.dimension}")
    
    # 테스트
    test_text = "Personal Health Agent는 무엇인가요?"
    embedding = embedder.embed_query(test_text)
    print(f"Query embedding shape: {len(embedding)}")
