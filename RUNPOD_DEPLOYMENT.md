# RunPod 배포 가이드 (Ollama + OpenWebUI)

이 문서는 gpt-oss-20b-PHA 프로젝트를 RunPod GPU 클라우드에 Ollama와 OpenWebUI를 사용하여 배포하는 방법을 설명합니다.

**💡 참고**: 이 가이드는 Docker 없이 직접 설치하는 방법을 다룹니다. OpenWebUI는 Python 패키지로 설치되어 더 가볍고 간단합니다.

## 📋 사전 준비

- RunPod 계정 (https://www.runpod.io/)
- 최소 $10 크레딧
- gpt-oss:20b 모델 (Ollama를 통해 자동 다운로드, 약 13GB)

## 🚀 단계별 배포 가이드

### 1단계: GPU Pod 생성

1. RunPod 대시보드 → **Pods** → **+ Deploy**
2. **GPU 선택** (권장):
   - **RTX 4090 24GB** (~$0.40-0.60/시간) - 권장
   - **A100 40GB** (~$1.20/시간) - 대용량
   - **RTX 3090 24GB** (~$0.25/시간) - 저렴
3. **Template**: "RunPod PyTorch" 또는 "PyTorch 2.x"
4. **Container Disk**: 60GB 이상 (모델 13GB + 시스템)
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

⚠️ **중요**: gpt-oss:20b 모델은 약 13GB이며, RunPod 네트워크 속도에 따라 5분~1시간 이상 소요될 수 있습니다.

```bash
# gpt-oss:20b 모델 다운로드 (약 13GB, 시간 소요)
ollama pull gpt-oss:20b

# 다운로드 상태 확인 (다른 터미널에서)
watch -n 5 'ollama list'

# 다운로드 완료 후 확인
ollama list

# 모델 테스트
ollama run gpt-oss:20b "안녕하세요. 간단한 테스트입니다."
```

**네트워크 속도 팁**:
- 다운로드 속도가 느린 경우 (< 5 MB/s), 잠시 기다리면 속도가 개선될 수 있습니다
- 다운로드 중 연결이 끊어지면 `ollama pull` 명령을 다시 실행하면 이어받기가 됩니다

**다른 모델 옵션:**
```bash
# Llama 3.1 8B (더 가벼움, 약 4.7GB)
ollama pull llama3.1:8b

# Mistral 7B (효율적, 약 4.1GB)
ollama pull mistral:7b

# Gemma 2 9B (구글, 약 5.4GB)
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

# 5. PyTorch 설치 (CUDA 버전에 맞게)
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (RunPod PyTorch 2.4 템플릿)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 6. 의존성 설치
pip install -r requirements.txt

# 7. OpenWebUI 설치 (Docker 없이 Python 패키지로 설치)
pip install open-webui
```

### 6단계: 환경 설정

```bash
# .env 파일이 있는지 확인
ls -la .env

# .env 파일이 없으면 생성 (보통 프로젝트에 이미 있음)
# 필요시 수정
nano .env
```

`.env` 파일 주요 내용:
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

# 성공 확인 (약 20-30MB)
du -sh chroma_db/
ls -la chroma_db/
```

### 8단계: OpenWebUI RAG Function 설정

```bash
# OpenWebUI에 RAG function 자동 등록
mkdir -p ~/.open-webui/functions
cp openwebui_rag_function.py ~/.open-webui/functions/

# 복사 확인
ls -la ~/.open-webui/functions/
```

### 9단계: OpenWebUI 시작 (Docker 없이)

#### 방법 1: 백그라운드 실행 (권장)

```bash
# 백그라운드로 OpenWebUI 시작
nohup open-webui serve --host 0.0.0.0 --port 8080 > openwebui.log 2>&1 &

# PID 확인
echo $!

# 로그 확인 (실시간)
tail -f openwebui.log

# 프로세스 확인
ps aux | grep open-webui

# 종료 시
pkill -f open-webui
```

#### 방법 2: screen 사용

```bash
# screen 세션 시작
screen -S openwebui

# OpenWebUI 실행
open-webui serve --host 0.0.0.0 --port 8080

# Detach: Ctrl+A, D
# Reattach: screen -r openwebui
# 종료: Ctrl+C 후 exit
```

#### 방법 3: 스크립트 사용

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
nohup open-webui serve --host 0.0.0.0 --port 8080 > openwebui.log 2>&1 &
echo "OpenWebUI started in background. Check logs with: tail -f openwebui.log"
EOF

# 실행 권한 부여
chmod +x start_openwebui.sh

# 실행
./start_openwebui.sh
```

### 10단계: 포트 노출 및 접속 URL 확인

#### 방법 1: RunPod 웹 대시보드 사용

1. **Pod 페이지** → **Edit Pod**
2. **Expose HTTP Ports** 섹션:
   - Port: `8080` 추가 (OpenWebUI)
   - Port: `11434` 추가 (Ollama API, 선택사항)
3. **Save**

#### 방법 2: RunPod API/CLI 사용

```bash
# RunPod API를 사용하여 포트 업데이트 (MCP 또는 API 클라이언트)
# Pod ID 확인: RunPod 대시보드에서 확인

# 예시: Pod ID가 "abc123xyz456" 인 경우
# API 호출로 포트 업데이트
```

**포트 노출 후 접속 URL**:

RunPod는 다음 형식으로 URL을 자동 생성합니다:

```
OpenWebUI: https://{POD_ID}-8080.proxy.runpod.net
Ollama API: https://{POD_ID}-11434.proxy.runpod.net

예시:
https://zevjg8llu8t7ko-8080.proxy.runpod.net (OpenWebUI)
https://zevjg8llu8t7ko-11434.proxy.runpod.net (Ollama API)
```

**POD_ID 확인 방법**:
- RunPod 대시보드의 Pod 이름 아래에 표시됩니다
- 또는 SSH 접속 시 호스트명에서 확인: `root@{POD_ID}:/workspace#`

### 11단계: 서비스 상태 확인

```bash
# Ollama 상태 확인
ps aux | grep ollama

# OpenWebUI 상태 확인
ps aux | grep open-webui

# 포트 확인
ss -tlnp | grep -E "(11434|8080)"

# 결과 예시:
# LISTEN 0  4096  127.0.0.1:11434  0.0.0.0:*  users:(("ollama",pid=779,fd=3))
# LISTEN 0  2048  0.0.0.0:8080     0.0.0.0:*  users:(("open-webui",pid=1557,fd=41))
```

### 12단계: OpenWebUI 사용 방법

1. **OpenWebUI 접속**: `https://{POD_ID}-8080.proxy.runpod.net`
2. **계정 생성**: 첫 접속 시 관리자 계정 생성
3. **모델 선택**: 
   - 채팅 화면 상단에서 `gpt-oss:20b` 선택
4. **RAG Function 활성화**:
   - 채팅 입력창 위의 **Functions** 버튼 클릭
   - `Personal Health Agent RAG` 또는 사용 가능한 RAG function 활성화
5. **질문하기**:
   ```
   Personal Health Agent의 주요 기능은 무엇인가요?
   ```

### 13단계: 테스트

#### Ollama API 테스트

```bash
# 로컬 테스트
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "안녕하세요",
  "stream": false
}'

# 외부 접속 테스트 (RunPod URL - 포트를 노출한 경우)
curl https://{POD_ID}-11434.proxy.runpod.net/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Hello",
  "stream": false
}'
```

#### OpenWebUI 테스트

브라우저에서 `https://{POD_ID}-8080.proxy.runpod.net` 접속

## 📊 모델 옵션

### 대용량 모델 (20GB+ VRAM)
```bash
ollama pull gpt-oss:20b  # 약 13GB
```

### 중형 모델 (16-24GB VRAM)
```bash
# Llama 3.1 8B
ollama pull llama3.1:8b  # 약 4.7GB

# Gemma 2 9B
ollama pull gemma2:9b  # 약 5.4GB

# Qwen 2.5 14B
ollama pull qwen2.5:14b  # 약 9GB
```

### 소형 모델 (8-12GB VRAM)
```bash
# Mistral 7B
ollama pull mistral:7b  # 약 4.1GB

# Llama 3.2 3B
ollama pull llama3.2:3b  # 약 2GB

# Phi-3 Mini
ollama pull phi3:mini  # 약 2.3GB
```

## 💰 비용 최적화

### 1. Spot 인스턴스 사용
- "Spot" 옵션 선택
- 약 50% 저렴
- 중단 가능성 있음 (작업 저장 주의)

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

# 로그 확인
tail -f ollama.log

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
watch -n 1 nvidia-smi
```

### 모델 다운로드 느림

```bash
# 다운로드 진행 상황 모니터링
watch -n 5 'ollama list'

# Ollama 모델 캐시 위치 확인
ls -la ~/.ollama/models

# 다운로드 중단 후 재시작 (이어받기 지원)
ollama pull gpt-oss:20b
```

### OpenWebUI 접속 안됨

```bash
# 로그 확인
tail -f openwebui.log

# 프로세스 확인
ps aux | grep open-webui

# 포트 확인
ss -tlnp | grep 8080

# 재시작
pkill -f open-webui
nohup open-webui serve --host 0.0.0.0 --port 8080 > openwebui.log 2>&1 &
```

### RAG Function 작동 안함

```bash
# Vector store 확인
ls -la chroma_db/
du -sh chroma_db/

# RAG function 파일 확인
ls -la ~/.open-webui/functions/

# 벡터 DB 재구축
python build_index.py

# Python 경로 확인
which python
python --version
```

### 포트가 노출되지 않음

```bash
# 포트 바인딩 확인
ss -tlnp | grep -E "(8080|11434)"

# 0.0.0.0으로 바인딩되어 있는지 확인 (외부 접속 가능)
# 127.0.0.1이면 localhost만 접속 가능

# OpenWebUI 재시작 (--host 0.0.0.0 확인)
pkill -f open-webui
nohup open-webui serve --host 0.0.0.0 --port 8080 > openwebui.log 2>&1 &
```

## 🔄 빠른 배포 스크립트

모든 단계를 자동화하는 스크립트 (Docker 없이):

```bash
#!/bin/bash
# deploy_runpod_no_docker.sh

set -e

echo "=== RunPod Ollama + OpenWebUI 배포 시작 (Docker 없이) ==="

# 1. Ollama 설치
echo "[1/9] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > ollama.log 2>&1 &
sleep 5

# 2. 모델 다운로드
echo "[2/9] Pulling model (this may take 5-60 minutes)..."
ollama pull gpt-oss:20b &
OLLAMA_PID=$!

# 3. 프로젝트 설정 (모델 다운로드와 병렬)
echo "[3/9] Cloning project..."
cd /workspace
if [ -d "gpt-oss-20b-pha-model" ]; then
  cd gpt-oss-20b-pha-model
  git pull
else
  git clone https://github.com/megaworks-dev/gpt-oss-20b-pha-model.git
  cd gpt-oss-20b-pha-model
fi

# 4. 가상환경 생성
echo "[4/9] Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# 5. 의존성 설치
echo "[5/9] Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install open-webui

# 6. 벡터 DB 구축
echo "[6/9] Building vector database..."
python build_index.py

# 7. OpenWebUI 함수 등록
echo "[7/9] Registering RAG function..."
mkdir -p ~/.open-webui/functions
cp openwebui_rag_function.py ~/.open-webui/functions/

# 8. 모델 다운로드 완료 대기
echo "[8/9] Waiting for model download to complete..."
wait $OLLAMA_PID

# 9. OpenWebUI 시작
echo "[9/9] Starting OpenWebUI..."
nohup open-webui serve --host 0.0.0.0 --port 8080 > openwebui.log 2>&1 &

# Pod ID 추출 시도
POD_ID=$(hostname)

echo ""
echo "==================================="
echo "✅ Deployment complete!"
echo "==================================="
echo ""
echo "Services running:"
echo "  Ollama API: http://localhost:11434"
echo "  OpenWebUI: http://localhost:8080"
echo ""
echo "External URLs (after exposing ports):"
echo "  OpenWebUI: https://${POD_ID}-8080.proxy.runpod.net"
echo "  Ollama API: https://${POD_ID}-11434.proxy.runpod.net"
echo ""
echo "⚠️  Don't forget to expose ports 8080 and 11434 in RunPod dashboard!"
echo ""
echo "Check logs:"
echo "  tail -f openwebui.log"
echo "  tail -f ollama.log"
```

사용:
```bash
chmod +x deploy_runpod_no_docker.sh
./deploy_runpod_no_docker.sh
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
6. **Docker 불필요**: OpenWebUI는 Python 패키지로 설치되어 Docker가 필요 없습니다

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
# Ollama를 localhost만 허용 (외부 접속 차단)
ollama serve --host 127.0.0.1:11434
```

## 📈 성능 모니터링

```bash
# GPU 사용률 모니터링 (실시간)
watch -n 1 nvidia-smi

# 메모리 사용량
free -h

# 디스크 사용량
df -h

# 프로세스 모니터링
htop

# 서비스 상태 확인 스크립트
cat > check_services.sh << 'EOF'
#!/bin/bash
echo "=== Service Status ==="
echo ""
echo "Ollama:"
ps aux | grep ollama | grep -v grep || echo "  Not running"
echo ""
echo "OpenWebUI:"
ps aux | grep open-webui | grep -v grep || echo "  Not running"
echo ""
echo "Ports:"
ss -tlnp | grep -E "(11434|8080)"
echo ""
echo "GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
EOF

chmod +x check_services.sh
./check_services.sh
```

## 🆚 Docker vs 직접 설치 비교

| 항목 | Docker 방식 | 직접 설치 (본 가이드) |
|------|-------------|----------------------|
| 설치 복잡도 | 중간 | 낮음 |
| 메모리 사용량 | 높음 (컨테이너 오버헤드) | 낮음 |
| 시작 시간 | 느림 | 빠름 |
| 디버깅 | 어려움 | 쉬움 |
| 의존성 관리 | 격리됨 | 가상환경 |
| GPU 접근 | nvidia-docker 필요 | 직접 접근 |
| 권장 용도 | 프로덕션 | 개발/테스트 |

**결론**: 빠른 배포와 테스트에는 직접 설치 방식이 더 적합합니다.
