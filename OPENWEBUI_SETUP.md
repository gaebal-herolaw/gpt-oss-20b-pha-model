# OpenWebUI 설정 가이드

## 1. OpenWebUI 서버 시작

### 방법 1: 배치 파일 사용 (권장)
```bash
start_openwebui.bat
```

### 방법 2: 직접 실행
```bash
.venv\Scripts\python.exe -m open_webui serve
```

서버가 시작되면 http://localhost:8080 에서 접속 가능합니다.

## 2. 초기 설정

1. 브라우저에서 **http://localhost:8080** 접속
2. **회원가입** 진행 (첫 사용자가 관리자가 됩니다)
   - 이메일: 임의의 이메일 입력 (예: admin@local)
   - 비밀번호: 원하는 비밀번호 설정

## 3. LM Studio 연결

1. OpenWebUI 우측 상단 **프로필 아이콘** 클릭
2. **Settings** → **Connections** 선택
3. **OpenAI API** 섹션에서:
   - **API Base URL**: `http://localhost:1234/v1`
   - **API Key**: `lm-studio` (아무 값이나 입력)
4. **Save** 클릭
5. 모델이 자동으로 감지되어 사용 가능해집니다

## 4. RAG Function 추가 (논문 검색 자동화)

1. OpenWebUI 우측 상단 **프로필 아이콘** 클릭
2. **Admin Panel** → **Functions** 선택
3. **Import Function** 또는 **Create New Function** 클릭
4. `openwebui_rag_function.py` 파일의 내용을 복사하여 붙여넣기
5. **Save** 클릭
6. Function을 **Enable** 상태로 전환

## 5. RAG 사용 방법

### 자동 모드 (Function 활성화 시)
- 질문을 입력하면 자동으로 논문을 검색하고 답변합니다
- 예: "Personal Health Agent의 주요 기능은 무엇인가요?"

### 수동 모드 (Function 비활성화 시)
- LM Studio 모델만 사용하여 답변합니다

## 6. 설정 조정

### RAG Function 설정
1. **Functions** 페이지에서 RAG Function 클릭
2. **Settings** 탭에서 조정 가능:
   - **TOP_K**: 검색할 문서 개수 (기본: 5)

### 모델 설정
1. 채팅 화면 상단의 **모델 선택** 드롭다운
2. LM Studio 모델 선택 (gpt-oss-20b)
3. 설정 아이콘 클릭하여 temperature, max_tokens 등 조정

## 7. 문제 해결

### LM Studio 연결 안됨
- LM Studio가 실행 중인지 확인
- LM Studio에서 모델이 로드되어 있는지 확인
- LM Studio Server가 시작되어 있는지 확인 (포트: 1234)

### RAG Function이 작동 안함
- Function이 Enable 상태인지 확인
- 벡터 데이터베이스가 구축되어 있는지 확인 (`chroma_db/` 폴더 존재)
- 콘솔에서 에러 메시지 확인

### 답변이 느림
- CPU 모드로 실행 중이면 느릴 수 있습니다
- LM Studio 설정에서 배치 크기 조정
- RAG Function의 TOP_K 값을 줄여보세요 (예: 3)

## 8. 추천 설정

**최적의 답변을 위해:**
- Temperature: 0.7
- Top-K Documents: 5
- Max Tokens: 2048

**빠른 응답을 위해:**
- Temperature: 0.5
- Top-K Documents: 3
- Max Tokens: 512
