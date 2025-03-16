# 1. CUDA 12.4와 Ubuntu 22.04를 기반으로 하는 베이스 이미지 사용
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 2. 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Python 3.11을 기본 python으로 설정 (심볼릭 링크 생성)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# 4. pip 최신 버전으로 업데이트
RUN python -m pip install --upgrade pip

# 5. uv 설치 (uv는 pip를 통해 설치 가능)
RUN pip install uv

# 6. 작업 디렉토리 설정 (open-r1 프로젝트 소스가 위치하는 곳)
WORKDIR /app

# 7. 현재 open-r1 폴더의 모든 파일을 컨테이너 내 /app으로 복사
COPY . .

# 7-1. Bash 셸 사용
SHELL ["/bin/bash", "-c"]

# 7-2. uv를 이용해 가상환경 생성 후 pip 업데이트 (한 RUN 명령어 내에서 실행)
RUN uv venv /venv --python 3.11 && source /venv/bin/activate && uv pip install --upgrade pip

# 8. uv를 통해 open‑r1의 dependencies 설치, 가상환경은 한 줄에서만 작동하므로 모든 줄에서 작동되게 환경 설정
ENV GIT_LFS_SKIP_SMUDGE=1
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
ENV PIP_ROOT_USER_ACTION=ignore

RUN uv pip install -e ".[dev]"
RUN uv pip install vllm==0.7.2
RUN uv pip install setuptools && uv pip install flash-attn --no-build-isolation

# 9. 컨테이너 시작 시 bash 쉘을 실행하도록 지정
CMD ["/bin/bash"]