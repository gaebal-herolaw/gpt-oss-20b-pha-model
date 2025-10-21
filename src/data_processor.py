from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from .config import Config

class DataProcessor:
    """마크다운 파일 로드 및 청크 분할"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Config.DATA_DIR
        
        # 텍스트 분할기 초기화
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=[
                "\n## ",      # H2 헤더
                "\n### ",     # H3 헤더
                "\n\n",       # 단락
                "\n",         # 줄바꿈
                " ",          # 공백
                ""
            ],
            length_function=len
        )
    
    def load_markdown_files(self) -> List[Document]:
        """마크다운 파일 로드"""
        documents = []
        
        # .md 파일 찾기
        md_files = list(self.data_dir.glob("*.md"))
        
        if not md_files:
            raise FileNotFoundError(f"No markdown files found in {self.data_dir}")
        
        print(f"Found {len(md_files)} markdown files")
        
        for md_file in md_files:
            print(f"Loading: {md_file.name}")
            
            # UTF-8로 읽기
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Document 객체 생성
            doc = Document(
                page_content=content,
                metadata={
                    "source": md_file.name,
                    "file_path": str(md_file),
                    "file_size": md_file.stat().st_size
                }
            )
            
            documents.append(doc)
        
        print(f"[OK] Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        print("Splitting documents into chunks...")
        
        chunks = self.splitter.split_documents(documents)
        
        # 각 청크에 ID 부여
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        
        print(f"[OK] Created {len(chunks)} chunks from {len(documents)} documents")
        
        # 통계 출력
        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
        print(f"  Average chunk size: {avg_chunk_size:.0f} characters")
        
        return chunks
    
    def process(self) -> List[Document]:
        """전체 프로세스 실행"""
        documents = self.load_markdown_files()
        chunks = self.split_documents(documents)
        return chunks

# 사용 예시
if __name__ == "__main__":
    processor = DataProcessor()
    chunks = processor.process()
    
    # 샘플 출력
    print("\n" + "=" * 60)
    print("Sample Chunk:")
    print("=" * 60)
    print(chunks[0].page_content[:500])
    print(f"\nMetadata: {chunks[0].metadata}")
