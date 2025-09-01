# NVIDIA CUDA + Python이 포함된 공식 베이스 이미지
# (CUDA 버전과 Python 버전은 프로젝트 라이브러리에 맞게 변경)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 기본 패키지 및 Python 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 & 설치
COPY requirements.txt .
# PyTorch CUDA 특정 버전 설치 위해 별도 설치 명령어
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir  \
    torch==2.6.0+cu124  \
    torchvision==0.21.0+cu124  \
    torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 프로젝트 소스 복사
COPY . .

# 컨테이너 실행 시 실행할 명령
#CMD ["python3"]
CMD ["python3", "run_model.py"]
