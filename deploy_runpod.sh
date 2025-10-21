#!/bin/bash

set -e

echo "========================================"
echo "RunPod 배포 자동화 스크립트"
echo "========================================"
echo ""

# 1. 프로젝트 클론
echo "[1/8] 프로젝트 클론..."
cd /workspace
if [ -d "gpt-oss-20b-pha-model" ]; then
    echo "기존 디렉토리 발견. 업데이트 중..."
    cd gpt-oss-20b-pha-model
    git pull
else
    git clone https://github.com/gaebal-herolaw/gpt-oss-20b-pha-model.git
    cd gpt-oss-20b-pha-model
fi

# 2. 가상환경 생성
echo ""
echo "[2/8] 가상환경 생성..."
python -m venv .venv
source .venv/bin/activate

# 3. CUDA 확인
echo ""
echo "[3/8] GPU 확인..."
nvidia-smi

# 4. PyTorch 설치
echo ""
echo "[4/8] PyTorch 설치..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 의존성 설치
echo ""
echo "[5/8] 의존성 설치..."
pip install -r requirements.txt

# 6. 모델 설정
echo ""
echo "[6/8] RunPod 모드로 설정..."
cp src/local_llm_runpod.py src/local_llm.py
cp .env.example .env

# GPU 활성화
sed -i 's/DEVICE = "cpu"/DEVICE = "cuda" if torch.cuda.is_available() else "cpu"/' src/config.py

echo "완료! .env 파일에서 모델을 선택하세요:"
echo "  nano .env"
echo "  LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2  # 권장"

# 7. 벡터 DB 구축
echo ""
echo "[7/8] 벡터 데이터베이스 구축..."
python build_index.py

# 8. API 서버 시작
echo ""
echo "[8/8] API 서버 시작..."
nohup python api_server.py > api.log 2>&1 &

echo ""
echo "========================================"
echo "배포 완료!"
echo "========================================"
echo ""
echo "API 서버가 백그라운드에서 실행 중입니다."
echo ""
echo "다음 단계:"
echo "1. RunPod에서 포트 8000을 노출하세요"
echo "2. 제공된 URL로 테스트하세요"
echo ""
echo "유용한 명령어:"
echo "  tail -f api.log          # 로그 확인"
echo "  ps aux | grep api_server # 프로세스 확인"
echo "  nvidia-smi               # GPU 사용량 확인"
echo ""
