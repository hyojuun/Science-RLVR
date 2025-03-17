# 사용자 이름과 가상환경 디렉토리를 변수로 선언
ARG USERNAME=hyojun
ARG VENV_DIR=/venv

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

# 6. 사용자 생성 (변수 USERNAME 사용)
ARG USERNAME
RUN useradd -ms /bin/bash ${USERNAME}

# 7. 작업 디렉토리 설정 및 코드 복사 (코드는 /app에 저장)
WORKDIR /app
COPY . .

# 8. 가상환경을 위한 디렉토리 생성 및 소유권 변경 (변수 VENV_DIR 사용)
ARG VENV_DIR
RUN mkdir -p ${VENV_DIR} && chown -R ${USERNAME}:${USERNAME} ${VENV_DIR} /app

# 9. 이제부터는 USERNAME으로 실행
USER ${USERNAME}

# 10. Bash 셸 사용
SHELL ["/bin/bash", "-c"]

# 11. uv를 이용해 가상환경 생성 후 pip 업데이트 (가상환경은 VENV_DIR에 생성)
RUN uv venv ${VENV_DIR} --python 3.11 && source ${VENV_DIR}/bin/activate && uv pip install --upgrade pip

# 12. 환경변수 설정 (가상환경은 VENV_DIR, 코드 폴더는 /app)
ENV GIT_LFS_SKIP_SMUDGE=1
ENV VIRTUAL_ENV=${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:$PATH"
ENV PIP_ROOT_USER_ACTION=ignore

# 13. 프로젝트 의존성 설치
RUN uv pip install -e ".[dev]"
RUN uv pip install vllm==0.7.2
RUN uv pip install setuptools && uv pip install flash-attn --no-build-isolation

# 14. 컨테이너 시작 시 bash 쉘 실행
CMD ["/bin/bash"]