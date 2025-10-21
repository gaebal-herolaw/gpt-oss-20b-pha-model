# gpt-oss-20b-PHA Model

RAG (Retrieval Augmented Generation) 기반 Personal Health Agent Q&A 시스템

## 🎯 프로젝트 개요

이 프로젝트는 LM Studio를 통해 gpt-oss-20b 모델을 사용하여 Personal Health Agent 연구 논문에 대한 질문-답변 시스템을 구현합니다.

### 주요 기능

- ✅ LM Studio 통합 및 OpenAI 호환 API
- ✅ 다국어 임베딩 지원 (한국어 + 영어)
- ✅ ChromaDB 벡터 스토어를 통한 효율적인 검색
- ✅ OpenWebUI 채팅 인터페이스 통합
- ✅ FastAPI REST API 서버
- ✅ 대화형 CLI 인터페이스
- ✅ 출처 추적 및 인용

## 🚀 빠른 시작

### 사전 요구사항

- Python 3.10 이상
- [LM Studio](https://lmstudio.ai/) 설치
- LM Studio에 gpt-oss-20b 모델 다운로드
- 16GB 이상 RAM (32GB 이상 권장)
- 20GB 이상 여유 디스크 공간

### 설치

```bash
# 저장소 클론
git clone https://github.com/gaebal-herolaw/gpt-oss-20b-pha-model.git
cd gpt-oss-20b-pha-model

# 가상환경 생성
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# PyTorch CUDA 지원 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성 설치
pip install -r requirements.txt

# .env 파일 생성
cp .env.example .env
```

### LM Studio 설정

1. **LM Studio 설치**: [https://lmstudio.ai/](https://lmstudio.ai/)에서 다운로드
2. **gpt-oss-20b 모델 다운로드** (기본 위치: `~/.lmstudio/models/`)
3. **LM Studio 서버 시작**:
   - gpt-oss-20b 모델 로드
   - "Developer" 또는 "Local Server" 탭으로 이동
   - "Start Server" 클릭 (기본 포트: 1234)

### 벡터 데이터베이스 구축

`data/` 폴더에 연구 논문 마크다운 파일을 추가한 후:

```bash
python build_index.py
```

`chroma_db/`에 ChromaDB 벡터 데이터베이스가 생성됩니다.

### 시스템 실행

#### 옵션 1: OpenWebUI (권장)

```bash
# OpenWebUI 시작
start_openwebui.bat
# 또는
.venv\Scripts\open-webui.exe serve
```

브라우저에서 http://localhost:8080 접속

자세한 설정은 [OPENWEBUI_SETUP.md](OPENWEBUI_SETUP.md)를 참조하세요.

#### 옵션 2: CLI 모드

```bash
python main.py
```

터미널에서 대화형 Q&A를 사용합니다.

#### 옵션 3: API 서버

```bash
python api_server.py
```

API는 http://localhost:8000 에서 사용 가능

Swagger UI: http://localhost:8000/docs

## 📁 프로젝트 구조

```
gpt-oss-20b-pha-model/
├── src/
│   ├── __init__.py
│   ├── config.py              # 설정
│   ├── embedding_model.py     # 임베딩 모델 (CPU)
│   ├── local_llm.py          # LM Studio API 클라이언트
│   ├── data_processor.py     # 데이터 처리
│   ├── vector_store.py       # ChromaDB 벡터 스토어
│   └── rag_chain.py          # RAG 파이프라인
├── data/                      # 연구 논문 (.md 파일)
├── chroma_db/                 # 벡터 데이터베이스 (자동 생성)
├── main.py                    # CLI 인터페이스
├── build_index.py            # 벡터 DB 빌더
├── api_server.py             # FastAPI 서버
├── test_search.py            # 벡터 검색 테스트
├── test_full_rag.py          # 전체 RAG 파이프라인 테스트
├── openwebui_rag_function.py # OpenWebUI RAG 함수
├── start_openwebui.bat       # OpenWebUI 실행 스크립트
├── requirements.txt          # 의존성
├── CLAUDE.md                 # 개발 가이드
├── OPENWEBUI_SETUP.md        # OpenWebUI 설정 가이드
└── README.md                 # 이 파일
```

## 🔧 설정

### .env 설정

`.env` 파일을 편집하여 커스터마이징:

```bash
# LM Studio 서버 URL
LM_STUDIO_URL=http://localhost:1234/v1

# 임베딩 모델
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# API 설정
API_HOST=0.0.0.0
API_PORT=8000
```

### 고급 설정

`src/config.py`를 편집하여 커스터마이징:

- 배치 크기 (기본값: 임베딩 64)
- 벡터 데이터베이스 파라미터
- 컨텍스트 길이 (기본값: 8192)
- 청크 크기 및 오버랩

**참고**: Blackwell GPU (sm_120)가 PyTorch 2.5.1에서 아직 지원되지 않아 현재 CPU 모드로 실행 중입니다. GPU 지원 시 `src/config.py`를 수정하세요:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 📚 사용 예시

### CLI 모드

```
질문: Personal Health Agent의 주요 기능은 무엇인가요?

[1/3] Searching for relevant documents...
[OK] Found 5 relevant chunks

[2/3] Creating prompt...
[OK] Prompt created (2341 characters)

[3/3] Generating answer...
[OK] Answer generated (856 characters)

답변:
Personal Health Agent는 다음과 같은 주요 기능을 제공합니다:
...
```

### API 모드

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Personal Health Agent의 주요 기능은 무엇인가요?",
    "k": 5,
    "temperature": 0.7,
    "max_length": 2048
  }'
```

### OpenWebUI 모드

1. http://localhost:8080 접속
2. `openwebui_rag_function.py`에서 RAG Function 추가
3. 채팅 인터페이스에서 질문
4. 시스템이 자동으로 논문을 검색하고 답변 생성

## 🧪 테스트

### 벡터 검색 테스트

```bash
python test_search.py
```

### 전체 RAG 파이프라인 테스트

```bash
python test_full_rag.py
```

## 🛠️ 문제 해결

### LM Studio 연결 안됨

```bash
# LM Studio 서버 실행 확인
curl http://localhost:1234/v1/models

# 예상 응답: 로드된 모델 목록
```

**해결 방법**:
- LM Studio가 실행 중인지 확인
- 모델이 로드되어 있는지 확인
- 서버가 1234 포트에서 시작되었는지 확인

### 벡터 데이터베이스를 찾을 수 없음

```bash
# 데이터베이스 재구축
python build_index.py
```

### 성능이 느림

**CPU 모드 최적화**:
- `src/config.py`에서 `EMBEDDING_BATCH_SIZE` 감소 (예: 64 → 32)
- `TOP_K_RESULTS` 감소 (예: 10 → 3)
- 생성 시 `max_length` 짧게 설정

### 메모리 문제

`src/config.py`에서 배치 크기 감소:

```python
EMBEDDING_BATCH_SIZE = 32  # 64에서 감소
GENERATION_BATCH_SIZE = 4  # 8에서 감소
```

## 📊 성능 벤치마크

| 작업 | 시간 | 메모리 사용량 |
|------|------|--------------|
| 임베딩 모델 로드 | 5-10초 | ~2GB RAM |
| 벡터 DB 구축 (1129 청크) | 4-5분 | ~5GB RAM |
| 쿼리 검색 | <1초 | ~0.5GB RAM |
| 답변 생성 (LM Studio) | 5-30초 | ~12GB RAM |

**참고**: 성능은 다음에 따라 달라집니다:
- LM Studio 모델 양자화 (MXFP4, Q4 등)
- 하드웨어 사양 (CPU/RAM)
- 벡터 DB의 문서 수

## 🆕 새로운 기능

### v2.0 (최신)
- ✅ LM Studio 통합 (직접 모델 로딩 대체)
- ✅ RAG 함수를 통한 OpenWebUI 지원
- ✅ 광범위한 호환성을 위한 CPU 모드
- ✅ Windows 인코딩 지원 개선
- ✅ 디버깅용 테스트 스크립트
- ✅ 포괄적인 문서 (CLAUDE.md, OPENWEBUI_SETUP.md)

### v1.0
- HuggingFace Transformers를 사용한 초기 릴리스
- 기본 RAG 파이프라인
- CLI 및 API 인터페이스

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.

## 🤝 기여

기여를 환영합니다! Pull Request를 자유롭게 제출해 주세요.

## 📮 문의

질문이나 이슈가 있으면 GitHub Issue를 열어주세요.

## 🔗 참고 자료

- [LM Studio](https://lmstudio.ai/)
- [OpenWebUI](https://github.com/open-webui/open-webui)
- [ChromaDB](https://www.trychroma.com/)
- [Personal Health Agent Paper](data/personal-health-agent.md)
