# Official CANN Dockerfile: Ascend/cann-container-image, 9.2.1-910b-ubuntu22.04-py3.12-devel.
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.1-910b-ubuntu22.04-py3.12-devel

ARG INSTALL_TEST_REQUIREMENTS=false
ARG PREWARM_THIRD_PARTY=true
ARG INSTALL_TORCH_NPU=true
ARG ASCEND_PYTORCH_RELEASE_VERSION=v26.1.0-beta.1
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1.post5
ARG TORCH_NPU_RELEASE_TAG=v26.1.0-beta.1-pytorch2.7.1
ARG TORCH_NPU_WHL=torch_npu-2.7.1.post5-cp312-cp312-manylinux_2_28_aarch64.whl
ARG TORCH_NPU_WHL_URL=
ARG TRITON_ASCEND_VERSION=3.2.1

# Inherit CANN paths and library paths from the official 9.2.1 image.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_VERSION=${TORCH_VERSION} \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    CI_LOG_CLEANUP_ENABLED=true \
    CI_LOG_RETENTION_DAYS=7

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      cmake \
      curl \
      g++ \
      gcc \
      git \
      git-lfs \
      make \
      ninja-build \
      openssh-client \
      patch \
      pkg-config \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      valgrind && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

COPY requirements.txt /tmp/fla-ci/requirements.txt

RUN set -e; \
    python3 -m pip install --upgrade pip setuptools wheel; \
    python3 -m pip install -r /tmp/fla-ci/requirements.txt; \
    if [[ "${INSTALL_TORCH_NPU}" == "true" ]]; then \
      case "${TORCH_NPU_RELEASE_TAG}" in \
        "${ASCEND_PYTORCH_RELEASE_VERSION}"-pytorch*) ;; \
        *) echo "TORCH_NPU_RELEASE_TAG must belong to ${ASCEND_PYTORCH_RELEASE_VERSION}: ${TORCH_NPU_RELEASE_TAG}" >&2; exit 2 ;; \
      esac; \
      python3 -m pip install "torch==${TORCH_VERSION}"; \
      torch_npu_url="${TORCH_NPU_WHL_URL:-https://gitcode.com/Ascend/pytorch/releases/download/${TORCH_NPU_RELEASE_TAG}/${TORCH_NPU_WHL}}"; \
      curl -fL --retry 3 --retry-delay 2 \
        -o "/tmp/${TORCH_NPU_WHL}" \
        "${torch_npu_url}"; \
      python3 -m pip install "/tmp/${TORCH_NPU_WHL}"; \
      rm -f "/tmp/${TORCH_NPU_WHL}"; \
      python3 -c 'from importlib import metadata; assert metadata.version("torch-npu") == "'"${TORCH_NPU_VERSION}"'"; print("torch-npu", metadata.version("torch-npu"))'; \
    fi; \
    python3 -m pip uninstall -y triton || true; \
    python3 -m pip install "triton-ascend==${TRITON_ASCEND_VERSION}" --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple; \
    python3 -m pip install --target=/tmp/pybind11-target pybind11; \
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])"); \
    cp -a /tmp/pybind11-target/. "$SITE_PACKAGES/"; \
    rm -rf /tmp/pybind11-target; \
    python3 -c "import pybind11; print(pybind11.__file__)"; \
    python3 -c "from triton import Config; print('triton Config OK')"

# Build and install ATK from source using the upstream README procedure.
# ATK 26.9.8 filesystem exchange names are incompatible with Kombu 5.7.
ARG ATK_COMMIT=0683181571486a6654571d7a433f43a4fe7a66c0
RUN set -e; \
    git init /opt/ATK; \
    git -C /opt/ATK remote add origin https://gitcode.com/Ascend/ATK.git; \
    git -C /opt/ATK -c http.version=HTTP/1.1 fetch --depth 1 origin "$ATK_COMMIT"; \
    git -C /opt/ATK checkout --detach FETCH_HEAD; \
    cd /opt/ATK; \
    python3 -m pip install wheel; \
    python3 setup.py bdist_wheel; \
    cd dist; \
    python3 -m pip install ./*.whl 'kombu==5.6.2'; \
    python3 -m pip show ATK; \
    command -v atk

RUN if [[ "${PREWARM_THIRD_PARTY}" == "true" ]]; then \
      mkdir -p /opt/fla-ci/third_party/pkg; \
      download() { \
        local url="$1"; \
        local dest="$2"; \
        if [[ -s "$dest" ]]; then return 0; fi; \
        echo "[CI][prewarm] downloading $url"; \
        if ! curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"; then \
          echo "[CI][prewarm][WARN] failed to download $url"; \
          rm -f "$dest"; \
        fi; \
      }; \
      clone_at() { \
        local url="$1"; \
        local rev="$2"; \
        local dest="$3"; \
        if [[ -d "$dest/.git" ]]; then return 0; fi; \
        echo "[CI][prewarm] cloning $url"; \
        if git clone "$url" "$dest" && git -C "$dest" checkout "$rev"; then \
          true; \
        else \
          echo "[CI][prewarm][WARN] failed to clone $url"; \
          rm -rf "$dest"; \
        fi; \
      }; \
      download "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz" \
        /opt/fla-ci/third_party/pkg/eigen-5.0.0.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz" \
        /opt/fla-ci/third_party/pkg/abseil-cpp-20230802.1.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz" \
        /opt/fla-ci/third_party/pkg/protobuf-25.1.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz" \
        /opt/fla-ci/third_party/pkg/makeself-release-2.5.0-patch1.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip" \
        /opt/fla-ci/third_party/pkg/include.zip; \
      clone_at "https://gitcode.com/cann/opbase.git" \
        c8cfc45e350d4e07cd8cab8448d3b40f9727ea4c \
        /opt/fla-ci/third_party/opbase; \
      clone_at "https://gitcode.com/cann/catlass.git" \
        41bf90da655bba3c66d0acd7e00abe33960ecfd6 \
        /opt/fla-ci/third_party/catlass; \
    fi

WORKDIR /workspace/repo
