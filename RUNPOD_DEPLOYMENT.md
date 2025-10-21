# RunPod 배포 가이드

이 문서는 gpt-oss-20b-PHA 프로젝트를 RunPod GPU 클라우드에 배포하는 방법을 설명합니다.

## 📋 사전 준비

- RunPod 계정 (https://www.runpod.io/)
- 최소 $10 크레딧
- gpt-oss-20b 모델 또는 대체 모델

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

또는 Web Terminal 사용

### 3단계: 프로젝트 설정

```bash
# 1. 작업 디렉토리로 이동
cd /workspace

# 2. 프로젝트 클론
git clone https://github.com/gaebal-herolaw/gpt-oss-20b-pha-model.git
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
```

### 4단계: RunPod용 모델 설정

```bash
# local_llm.py를 RunPod 버전으로 교체
cp src/local_llm.py src/local_llm_lmstudio.py  # 백업
cp src/local_llm_runpod.py src/local_llm.py

# .env 파일 수정
nano .env
```

`.env` 파일 내용:
```bash
# GPU 설정
CUDA_VISIBLE_DEVICES=0

# 모델 경로
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# LLM 모델 (다음 중 선택)
# 옵션 1: gpt-oss-20b (20B, 40GB+ VRAM 필요)
LLM_MODEL=gpt-oss-20b

# 옵션 2: Llama-2 7B (7B, 14GB VRAM 필요)
# LLM_MODEL=meta-llama/Llama-2-7b-chat-hf

# 옵션 3: Mistral 7B (7B, 14GB VRAM 필요)
# LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# API 설정
API_HOST=0.0.0.0
API_PORT=8000
```

### 5단계: GPU 활성화

```bash
# src/config.py 수정
nano src/config.py
```

다음 라인 수정:
```python
# 기존:
# DEVICE = "cpu"

# 수정:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### 6단계: 벡터 데이터베이스 구축

```bash
# 데이터가 이미 있는 경우
python build_index.py

# 성공 확인
ls -la chroma_db/
```

### 7단계: API 서버 시작

#### 방법 1: 백그라운드 실행 (권장)

```bash
# nohup으로 백그라운드 실행
nohup python api_server.py > api.log 2>&1 &

# 로그 확인
tail -f api.log

# PID 확인
ps aux | grep api_server

# 종료 시
kill <PID>
```

#### 방법 2: screen 사용

```bash
# screen 세션 시작
screen -S api

# API 서버 실행
python api_server.py

# Detach: Ctrl+A, D
# Reattach: screen -r api
# 종료: Ctrl+C 후 exit
```

### 8단계: 포트 노출

RunPod Pod 설정:

1. **Pod 페이지** → **Edit Pod**
2. **Expose HTTP Ports** 섹션:
   - Port: `8000` 추가
3. **Save**

Pod URL 확인:
```
https://xxxxx-8000.proxy.runpod.net
```

### 9단계: 테스트

```bash
# Health 체크
curl https://xxxxx-8000.proxy.runpod.net/health

# 질문 테스트
curl -X POST "https://xxxxx-8000.proxy.runpod.net/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Personal Health Agent의 주요 기능은 무엇인가요?",
    "k": 5,
    "temperature": 0.7,
    "max_length": 512
  }'
```

## 📊 모델 옵션

### 대용량 모델 (40GB+ VRAM)
```bash
LLM_MODEL=gpt-oss-20b
```

### 중형 모델 (24GB VRAM)
```bash
# Llama-2 13B
LLM_MODEL=meta-llama/Llama-2-13b-chat-hf

# Mistral 7B (효율적)
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

### 소형 모델 (12GB VRAM)
```bash
# Llama-2 7B
LLM_MODEL=meta-llama/Llama-2-7b-chat-hf

# Phi-2 (매우 효율적)
LLM_MODEL=microsoft/phi-2
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
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

## 🛠️ 문제 해결

### GPU 메모리 부족 (OOM)

**해결 1: 양자화 사용**
```python
# src/local_llm_runpod.py의 load_kwargs에 추가
load_kwargs["load_in_8bit"] = True
```

**해결 2: 작은 모델 사용**
```bash
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**해결 3: 배치 크기 감소**
```python
# src/config.py
EMBEDDING_BATCH_SIZE = 32  # 64에서 감소
```

### 모델 다운로드 느림

```bash
# HuggingFace 캐시 사용
export HF_HOME=/workspace/.cache/huggingface

# 또는 미리 다운로드
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
```

### API 서버 응답 없음

```bash
# 로그 확인
tail -f api.log

# 프로세스 확인
ps aux | grep python

# 포트 확인
netstat -tlnp | grep 8000
```

## 🔄 배포 스크립트

모든 단계를 자동화하는 스크립트:

```bash
# deploy_runpod.sh
#!/bin/bash

set -e

echo "=== RunPod 배포 시작 ==="

# 1. 프로젝트 클론
cd /workspace
git clone https://github.com/gaebal-herolaw/gpt-oss-20b-pha-model.git
cd gpt-oss-20b-pha-model

# 2. 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 3. 의존성 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. 모델 설정
cp src/local_llm_runpod.py src/local_llm.py
cp .env.example .env

# 5. GPU 활성화
sed -i 's/DEVICE = "cpu"/DEVICE = "cuda" if torch.cuda.is_available() else "cpu"/' src/config.py

# 6. 벡터 DB 구축
python build_index.py

# 7. API 서버 시작
nohup python api_server.py > api.log 2>&1 &

echo "=== 배포 완료! ==="
echo "API URL: https://xxxxx-8000.proxy.runpod.net"
```

사용:
```bash
chmod +x deploy_runpod.sh
./deploy_runpod.sh
```

## 📚 추가 리소스

- [RunPod 공식 문서](https://docs.runpod.io/)
- [HuggingFace 모델 허브](https://huggingface.co/models)
- [프로젝트 GitHub](https://github.com/gaebal-herolaw/gpt-oss-20b-pha-model)

## ⚠️ 주의사항

1. **비용 관리**: 사용하지 않을 때는 Pod 중지
2. **데이터 백업**: Volume에 중요 데이터 저장
3. **모델 라이선스**: 사용하는 모델의 라이선스 확인
4. **VRAM 모니터링**: `nvidia-smi` 명령어로 확인
