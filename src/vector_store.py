import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from pathlib import Path
from .config import Config
from .embedding_model import LocalEmbedding

class VectorStore:
    """ChromaDB 기반 벡터 스토어"""
    
    def __init__(
        self,
        persist_dir: Path = None,
        collection_name: str = "pha_papers",
        embedder: LocalEmbedding = None
    ):
        self.persist_dir = persist_dir or Config.CHROMA_DB_DIR
        self.collection_name = collection_name
        self.embedder = embedder or LocalEmbedding()
        
        # ChromaDB 클라이언트 초기화
        print(f"Initializing ChromaDB at {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = None
    
    def create_collection(self, reset: bool = False):
        """컬렉션 생성"""
        
        # 기존 컬렉션 삭제 (reset=True인 경우)
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"✓ Deleted existing collection: {self.collection_name}")
            except:
                pass
        
        # 새 컬렉션 생성
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Personal Health Agent Research Papers",
                "embedding_model": self.embedder.model_name
            }
        )
        
        print(f"✓ Collection ready: {self.collection_name}")
        print(f"  Total documents: {self.collection.count()}")
    
    def add_documents(self, chunks: List[Any]):
        """문서 추가"""
        
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        # 텍스트와 메타데이터 추출
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        print(f"\nAdding {len(texts)} documents to vector store...")
        
        # 임베딩 생성 (GPU에서 배치 처리)
        print("Generating embeddings...")
        embeddings = self.embedder.embed_documents(texts)
        
        # ChromaDB에 배치 추가
        print("Adding to ChromaDB...")
        batch_size = Config.VECTOR_DB_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            print(f"  Progress: {end_idx}/{len(texts)} ({end_idx/len(texts)*100:.1f}%)")
        
        print(f"✓ Added {len(texts)} documents to vector store")
    
    def search(
        self,
        query: str,
        k: int = None,
        filter_dict: Dict = None
    ) -> Dict[str, Any]:
        """유사 문서 검색"""
        
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        k = k or Config.TOP_K_RESULTS
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedder.embed_query(query)
        
        # 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict  # 메타데이터 필터링
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        if self.collection is None:
            return {"error": "Collection not initialized"}
        
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_dimension": self.embedder.dimension,
            "embedding_model": self.embedder.model_name
        }

# 사용 예시
if __name__ == "__main__":
    from .data_processor import DataProcessor
    
    # 데이터 처리
    processor = DataProcessor()
    chunks = processor.process()
    
    # 벡터 스토어 생성
    vector_store = VectorStore()
    vector_store.create_collection(reset=True)
    vector_store.add_documents(chunks)
    
    # 검색 테스트
    results = vector_store.search("Personal Health Agent의 주요 기능")
    print("\n" + "=" * 60)
    print("Search Results:")
    print("=" * 60)
    print(f"Query: Personal Health Agent의 주요 기능")
    print(f"Found: {len(results['documents'][0])} results")
    print(f"\nTop result:\n{results['documents'][0][0][:300]}...")
