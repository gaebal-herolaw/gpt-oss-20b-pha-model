import torch
from pathlib import Path

class Config:
    # 프로젝트 경로
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # GPU 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # RTX Pro 6000 96GB 최적화 설정
    EMBEDDING_BATCH_SIZE = 64      # 대용량 배치
    GENERATION_BATCH_SIZE = 8
    MAX_CONTEXT_LENGTH = 8192      # 긴 컨텍스트
    
    # 모델 설정
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    LLM_MODEL = "gpt-oss-20b"  # 실제 모델 경로로 변경
    MODEL_DTYPE = torch.float16
    
    # 벡터 DB 설정
    VECTOR_DB_BATCH_SIZE = 2000
    TOP_K_RESULTS = 10
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # API 설정
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    @staticmethod
    def print_gpu_info():
        """GPU 정보 출력"""
        if torch.cuda.is_available():
            print("=" * 60)
            print("GPU Information")
            print("=" * 60)
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print("=" * 60)
        else:
            print("CUDA is not available. Using CPU.")
    
    @staticmethod
    def create_directories():
        """필요한 디렉토리 생성"""
        Config.CHROMA_DB_DIR.mkdir(exist_ok=True)
        Config.MODELS_DIR.mkdir(exist_ok=True)
        Config.DATA_DIR.mkdir(exist_ok=True)
        print("✓ Directories created")

# 초기화
Config.create_directories()
