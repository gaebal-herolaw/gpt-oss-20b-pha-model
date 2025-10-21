# RunPod 배포 가이드 (Ollama + OpenWebUI)

이 문서는 gpt-oss-20b-PHA 프로젝트를 RunPod GPU 클라우드에 Ollama와 OpenWebUI를 사용하여 배포하는 방법을 설명합니다.

## 📋 사전 준비

- RunPod 계정 (https://www.runpod.io/)
- 최소 $10 크레딧
- gpt-oss:20b 모델 (Ollama를 통해 자동 다운로드)

## 🚀 단계별 배포 가이드

### 1단계: GPU Pod 생성

1. RunPod 대시보드 → **Pods** → **+ Deploy**
2. **GPU 선택** (권장):
   - **RTX 4090 24GB** (~$0.40/시간) - 권장
   - **A100 40GB** (~$1.20/시간) - 대용량
   - **RTX 3090 24GB** (~$0.25/시간) - 저렴
3. **Template**: "RunPod PyTorch" 또는 "PyTorch 2.x"
4. **Container Disk**: 50GB 이상
5. **Volume Disk**: 100GB (선택, 데이터 영구 보관)
6. **Deploy On-Demand** 클릭

### 2단계: SSH 접속

Pod 생성 후 (1-2분):

```bash
# RunPod에서 제공하는 SSH 명령어 사용
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

또는 Web Terminal 또는 JupyterLab Terminal 사용

### 3단계: Ollama 설치

```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# Ollama 서비스 시작 (백그라운드)
nohup ollama serve > ollama.log 2>&1 &

# 설치 확인
ollama --version
```

### 4단계: 모델 다운로드

```bash
# gpt-oss:20b 모델 다운로드
ollama pull gpt-oss:20b

# 다운로드 확인
ollama list

# 모델 테스트
ollama run gpt-oss:20b "안녕하세요"
```

**다른 모델 옵션:**
```bash
# Llama 3.1 8B (더 가벼움)
ollama pull llama3.1:8b

# Mistral 7B (효율적)
ollama pull mistral:7b

# Gemma 2 9B (구글)
ollama pull gemma2:9b
```

### 5단계: 프로젝트 설정

```bash
# 1. 작업 디렉토리로 이동
cd /workspace

# 2. 프로젝트 클론
git clone https://github.com/megaworks-dev/gpt-oss-20b-pha-model.git
cd gpt-oss-20b-pha-model

# 3. 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 4. CUDA 버전 확인
nvidia-smi

# 5. PyTorch 설치 (CUDA 12.1 예시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. 의존성 설치
pip install -r requirements.txt

# 7. OpenWebUI 설치
pip install open-webui
```

### 6단계: 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 수정
nano .env
```

`.env` 파일 내용:
```bash
# GPU 설정
CUDA_VISIBLE_DEVICES=0

# 임베딩 모델
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Ollama 설정
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# Vector DB Settings
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=pha_papers

# Generation Settings
MAX_LENGTH=2048
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
```

### 7단계: 벡터 데이터베이스 구축

```bash
# 임베딩 모델 다운로드 및 인덱스 구축
python build_index.py

# 성공 확인
ls -la chroma_db/
```

### 8단계: OpenWebUI 설정

#### OpenWebUI RAG Function 등록

1. OpenWebUI 실행 후 웹 인터페이스 접속
2. **Settings** → **Functions** → **+ Add Function**
3. `openwebui_rag_function.py` 내용 복사 & 붙여넣기
4. **Save** 클릭

또는 자동 등록:

```bash
# OpenWebUI에 RAG function 복사
mkdir -p ~/.open-webui/functions
cp openwebui_rag_function.py ~/.open-webui/functions/
```

### 9단계: 서비스 시작

#### 방법 1: 스크립트 사용 (Linux/Mac)

```bash
# start_openwebui.sh 생성
cat > start_openwebui.sh << 'EOF'
#!/bin/bash
cd /workspace/gpt-oss-20b-pha-model
source .venv/bin/activate
echo "Starting OpenWebUI..."
echo ""
echo "OpenWebUI will be available at: http://localhost:8080"
echo ""
open-webui serve --host 0.0.0.0 --port 8080
EOF

# 실행 권한 부여
chmod +x start_openwebui.sh

# 실행
./start_openwebui.sh
```

#### 방법 2: 백그라운드 실행

```bash
# 백그라운드로 OpenWebUI 시작
nohup open-webui serve --host 0.0.0.0 --port 8080 > openwebui.log 2>&1 &

# 로그 확인
tail -f openwebui.log

# PID 확인
ps aux | grep open-webui

# 종료 시
kill <PID>
```

#### 방법 3: screen 사용

```bash
# screen 세션 시작
screen -S openwebui

# OpenWebUI 실행
open-webui serve --host 0.0.0.0 --port 8080

# Detach: Ctrl+A, D
# Reattach: screen -r openwebui
# 종료: Ctrl+C 후 exit
```

### 10단계: 포트 노출

RunPod Pod 설정:

1. **Pod 페이지** → **Edit Pod**
2. **Expose HTTP Ports** 섹션:
   - Port: `8080` 추가 (OpenWebUI)
   - Port: `11434` 추가 (Ollama API, 선택사항)
3. **Save**

Pod URL 확인:
```
OpenWebUI: https://xxxxx-8080.proxy.runpod.net
Ollama API: https://xxxxx-11434.proxy.runpod.net
```

### 11단계: 사용 방법

1. **OpenWebUI 접속**: `https://xxxxx-8080.proxy.runpod.net`
2. **계정 생성**: 첫 접속 시 관리자 계정 생성
3. **모델 선택**: 
   - 채팅 화면 상단에서 `gpt-oss:20b` 선택
4. **RAG Function 활성화**:
   - 채팅 입력창 위의 **Functions** 버튼 클릭
   - `Personal Health Agent RAG` 활성화
5. **질문하기**:
   ```
   Personal Health Agent의 주요 기능은 무엇인가요?
   ```

### 12단계: 테스트

#### Ollama API 테스트

```bash
# 직접 테스트
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "안녕하세요",
  "stream": false
}'

# 외부 접속 테스트 (RunPod URL)
curl https://xxxxx-11434.proxy.runpod.net/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Hello",
  "stream": false
}'
```

#### OpenWebUI 테스트

브라우저에서 `https://xxxxx-8080.proxy.runpod.net` 접속

## 📊 모델 옵션

### 대용량 모델 (40GB+ VRAM)
```bash
ollama pull gpt-oss:20b
```

### 중형 모델 (16-24GB VRAM)
```bash
# Llama 3.1 8B
ollama pull llama3.1:8b

# Gemma 2 9B
ollama pull gemma2:9b

# Qwen 2.5 14B
ollama pull qwen2.5:14b
```

### 소형 모델 (8-12GB VRAM)
```bash
# Mistral 7B
ollama pull mistral:7b

# Llama 3.2 3B
ollama pull llama3.2:3b

# Phi-3 Mini
ollama pull phi3:mini
```

## 💰 비용 최적화

### 1. Spot 인스턴스 사용
- "Spot" 옵션 선택
- 약 50% 저렴
- 중단 가능성 있음

### 2. Pod 중지
```bash
# 사용하지 않을 때 RunPod 대시보드에서 Pod 중지
# 데이터는 Volume에 저장되어 유지됨
```

### 3. 작은 모델 사용
```bash
# Mistral 7B 추천 (성능 좋고 효율적)
ollama pull mistral:7b
```

### 4. 양자화 모델 사용
```bash
# 4-bit 양자화 (VRAM 절약)
ollama pull gpt-oss:20b-q4

# 8-bit 양자화
ollama pull gpt-oss:20b-q8
```

## 🛠️ 문제 해결

### Ollama 서비스 시작 안됨

```bash
# Ollama 프로세스 확인
ps aux | grep ollama

# 수동으로 시작
ollama serve

# 또는 백그라운드
nohup ollama serve > ollama.log 2>&1 &
```

### GPU 메모리 부족 (OOM)

**해결 1: 양자화 모델 사용**
```bash
ollama pull gpt-oss:20b-q4
```

**해결 2: 작은 모델 사용**
```bash
ollama pull mistral:7b
```

**해결 3: GPU 메모리 확인**
```bash
nvidia-smi
```

### 모델 다운로드 느림

```bash
# Ollama 모델 캐시 위치 확인
ls -la ~/.ollama/models

# 다운로드 상태 확인
ollama list
```

### OpenWebUI 접속 안됨

```bash
# 로그 확인
tail -f openwebui.log

# 프로세스 확인
ps aux | grep open-webui

# 포트 확인
netstat -tlnp | grep 8080

# 재시작
pkill -f open-webui
open-webui serve --host 0.0.0.0 --port 8080
```

### RAG Function 작동 안함

```bash
# Vector store 확인
ls -la chroma_db/

# 재구축
python build_index.py

# Python 경로 확인
which python
python --version
```

## 🔄 자동 배포 스크립트

모든 단계를 자동화하는 스크립트:

```bash
# deploy_runpod_ollama.sh
#!/bin/bash

set -e

echo "=== RunPod Ollama + OpenWebUI 배포 시작 ==="

# 1. Ollama 설치
echo "[1/10] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > ollama.log 2>&1 &
sleep 5

# 2. 모델 다운로드
echo "[2/10] Pulling model..."
ollama pull gpt-oss:20b

# 3. 프로젝트 클론
echo "[3/10] Cloning project..."
cd /workspace
git clone https://github.com/megaworks-dev/gpt-oss-20b-pha-model.git
cd gpt-oss-20b-pha-model

# 4. 가상환경 생성
echo "[4/10] Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# 5. 의존성 설치
echo "[5/10] Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install open-webui

# 6. 환경 설정
echo "[6/10] Setting up environment..."
cp .env.example .env
sed -i 's/LLM_MODEL=gpt-oss-20b/OLLAMA_MODEL=gpt-oss:20b/' .env
echo "OLLAMA_HOST=http://localhost:11434" >> .env

# 7. 벡터 DB 구축
echo "[7/10] Building vector database..."
python build_index.py

# 8. OpenWebUI 함수 등록
echo "[8/10] Registering RAG function..."
mkdir -p ~/.open-webui/functions
cp openwebui_rag_function.py ~/.open-webui/functions/

# 9. OpenWebUI 시작
echo "[9/10] Starting OpenWebUI..."
nohup open-webui serve --host 0.0.0.0 --port 8080 > openwebui.log 2>&1 &

echo "[10/10] Deployment complete!"
echo ""
echo "==================================="
echo "Ollama API: http://localhost:11434"
echo "OpenWebUI: http://localhost:8080"
echo "==================================="
echo ""
echo "External URLs (after port exposure):"
echo "OpenWebUI: https://xxxxx-8080.proxy.runpod.net"
echo "Ollama API: https://xxxxx-11434.proxy.runpod.net"
```

사용:
```bash
chmod +x deploy_runpod_ollama.sh
./deploy_runpod_ollama.sh
```

## 🎨 OpenWebUI 고급 설정

### 사용자 정의 프롬프트

OpenWebUI → **Settings** → **Prompts**:

```
당신은 Personal Health Agent 연구 논문 전문가입니다.
다음 논문 내용을 바탕으로 정확하고 상세하게 답변해주세요:

{context}

질문: {query}

답변 시 주의사항:
1. 논문 내용을 기반으로만 답변하세요
2. 출처를 명확히 표시하세요
3. 불확실한 내용은 언급하지 마세요
```

### 모델 파라미터 조정

OpenWebUI → 모델 선택 → **Settings**:

- **Temperature**: 0.7 (창의성)
- **Top P**: 0.9 (다양성)
- **Top K**: 50 (선택 범위)
- **Max Tokens**: 2048 (응답 길이)

## 📚 추가 리소스

- [Ollama 공식 문서](https://github.com/ollama/ollama)
- [OpenWebUI 공식 문서](https://docs.openwebui.com/)
- [RunPod 공식 문서](https://docs.runpod.io/)
- [프로젝트 GitHub](https://github.com/megaworks-dev/gpt-oss-20b-pha-model)

## ⚠️ 주의사항

1. **비용 관리**: 사용하지 않을 때는 Pod 중지
2. **데이터 백업**: Volume에 중요 데이터 저장
3. **모델 라이선스**: 사용하는 모델의 라이선스 확인
4. **VRAM 모니터링**: `nvidia-smi` 명령어로 확인
5. **포트 보안**: 프로덕션 환경에서는 인증 설정 필수

## 🔐 보안 권장사항

### OpenWebUI 인증 설정

```bash
# 환경 변수로 기본 사용자 설정
export WEBUI_AUTH=True
export WEBUI_NAME="PHA Research"
export WEBUI_SECRET_KEY="your-secret-key-here"

# OpenWebUI 시작
open-webui serve --host 0.0.0.0 --port 8080
```

### Ollama API 제한

```bash
# Ollama를 localhost만 허용
ollama serve --host 127.0.0.1:11434
```

## 📈 성능 모니터링

```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 메모리 사용량
free -h

# 디스크 사용량
df -h

# 프로세스 모니터링
htop
```
