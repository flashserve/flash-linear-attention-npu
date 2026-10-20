FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel

ARG INSTALL_TEST_REQUIREMENTS=false
ARG PREWARM_THIRD_PARTY=true
ARG INSTALL_TORCH_NPU=true
ARG ASCEND_PYTORCH_RELEASE_VERSION=v26.1.0-beta.1
ARG TORCH_VERSION=2.7.1
# Optional explicit torch wheel.  Left empty the arch decides the source:
# aarch64 installs "torch==<version>" from the index (arm64 wheels are CPU-only
# either way), while x86_64 takes the official +cpu wheel, because torch_npu
# x86 pins "torch==<version>+cpu" and the plain PyPI wheel there is the CUDA
# build (~3 GB of nvidia-* packages an Ascend host never uses).
ARG TORCH_WHL=
ARG TORCH_WHL_URL=
ARG TORCH_WHL_SHA256=
ARG TORCH_NPU_VERSION=2.7.1.post5
ARG TORCH_NPU_RELEASE_TAG=v26.1.0-beta.1-pytorch2.7.1
# Empty default: the wheel is picked from the build host's arch (aarch64 /
# x86_64), so one recipe serves both arches and neither can silently pick the
# other's payload.  Override TORCH_NPU_WHL to pin an exact file.
ARG TORCH_NPU_WHL=
ARG TORCH_NPU_WHL_URL=
ARG TRITON_ASCEND_VERSION=3.2.1
# Optional PyPI mirror for build hosts that cannot reach pypi.org quickly; empty
# keeps the default index.
ARG PYPI_INDEX_URL=
# Optional apt mirror for build hosts that cannot reach archive.ubuntu.com
# (the usual case on the office network); empty keeps the image's own sources.
ARG APT_MIRROR=

ENV DEBIAN_FRONTEND=noninteractive \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_VERSION=${TORCH_VERSION} \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    CI_LOG_CLEANUP_ENABLED=true \
    CI_LOG_RETENTION_DAYS=7 \
    FLA_NPU_CI_BASELINE="cann=9.1.0-910b;ubuntu=22.04;gcc=11;torch=${TORCH_VERSION};torch_npu=${TORCH_NPU_VERSION};triton_ascend=${TRITON_ASCEND_VERSION}"

SHELL ["/bin/bash", "-lc"]

RUN if [[ -n "${APT_MIRROR}" ]]; then \
      sed -i -e "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
             -e "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
             -e "s|http://ports.ubuntu.com/ubuntu-ports|${APT_MIRROR}|g" \
             /etc/apt/sources.list \
             /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
      echo "[fla-ci] apt mirror: ${APT_MIRROR}"; \
    fi; \
    apt-get update && \
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

# The wheels promise manylinux_2_34 / GLIBCXX <= 3.4.29, and that watermark is a
# property of this image's toolchain, not of the wheel build.  Fail the image
# build when the base drifts to another gcc instead of letting the promise break
# on a user's host later.
RUN set -e; \
    gcc_major="$(gcc -dumpversion | cut -d. -f1)"; \
    echo "[fla-ci] arch=$(uname -m) release=$(. /etc/os-release; echo "$PRETTY_NAME") gcc=${gcc_major} ($(gcc --version | head -1))"; \
    test "${gcc_major}" = "11" || { \
      echo "[fla-ci][FATAL] this image must build with the gcc 11 toolchain: the wheel carries a GLIBCXX_3.4.29 watermark that gcc ${gcc_major} raises to $(gcc -dumpversion)" >&2; \
      exit 1; \
    }

LABEL org.fla-npu.ci.baseline="cann=9.1.0-910b;ubuntu=22.04;gcc=11" \
      org.fla-npu.ci.purpose="flash-linear-attention-npu wheel builds (arm64 and amd64 share this recipe)"

COPY requirements.txt /tmp/fla-ci/requirements.txt

# Empty TORCH_WHL_URL picks the source by arch: x86_64 must take the official
# torch==<version>+cpu wheel (the x86 torch_npu pins that exact local version and
# the plain PyPI wheel there is the CUDA build), aarch64 installs
# torch==<version> (arm64 wheels are CPU-only either way).  One recipe, two
# arches, and neither can silently install the other's payload.
RUN set -e; \
    if [[ -n "${PYPI_INDEX_URL}" ]]; then export PIP_INDEX_URL="${PYPI_INDEX_URL}"; fi; \
    python3 -m pip install --upgrade pip setuptools wheel; \
    python3 -m pip install -r /tmp/fla-ci/requirements.txt; \
    if [[ "${INSTALL_TORCH_NPU}" == "true" ]]; then \
      case "${TORCH_NPU_RELEASE_TAG}" in \
        "${ASCEND_PYTORCH_RELEASE_VERSION}"-pytorch*) ;; \
        *) echo "TORCH_NPU_RELEASE_TAG must belong to ${ASCEND_PYTORCH_RELEASE_VERSION}: ${TORCH_NPU_RELEASE_TAG}" >&2; exit 2 ;; \
      esac; \
      torch_whl="${TORCH_WHL}"; \
      torch_whl_url="${TORCH_WHL_URL}"; \
      torch_whl_sha="${TORCH_WHL_SHA256}"; \
      if [[ -z "${torch_whl_url}" && "$(uname -m)" == "x86_64" ]]; then \
        torch_whl="torch-${TORCH_VERSION}+cpu-cp312-cp312-manylinux_2_28_x86_64.whl"; \
        torch_whl_url="https://download-r2.pytorch.org/whl/cpu/torch-${TORCH_VERSION}%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"; \
        if [[ "${TORCH_VERSION}" == "2.7.1" ]]; then \
          torch_whl_sha="8f8b3cfc53010a4b4a3c7ecb88c212e9decc4f5eeb6af75c3c803937d2d60947"; \
        fi; \
      fi; \
      if [[ -n "${torch_whl_url}" ]]; then \
        echo "[fla-ci] torch wheel: $(uname -m) ${torch_whl_url}"; \
        curl --http1.1 -fL --retry 5 --retry-delay 2 --connect-timeout 15 -o "/tmp/${torch_whl}" "${torch_whl_url}"; \
        if [[ -n "${torch_whl_sha}" ]]; then \
          echo "${torch_whl_sha}  /tmp/${torch_whl}" | sha256sum -c -; \
        else \
          echo "[fla-ci][WARN] no pinned sha256 for ${torch_whl}; actual: $(sha256sum /tmp/${torch_whl} | cut -d' ' -f1)"; \
        fi; \
        python3 -m pip install "/tmp/${torch_whl}"; \
        rm -f "/tmp/${torch_whl}"; \
      else \
        python3 -m pip install "torch==${TORCH_VERSION}"; \
      fi; \
      torch_npu_whl="${TORCH_NPU_WHL:-torch_npu-${TORCH_NPU_VERSION}-cp312-cp312-manylinux_2_28_$(uname -m).whl}"; \
      torch_npu_url="${TORCH_NPU_WHL_URL:-https://gitcode.com/Ascend/pytorch/releases/download/${TORCH_NPU_RELEASE_TAG}/${torch_npu_whl}}"; \
      curl -fL --retry 3 --retry-delay 2 \
        -o "/tmp/${torch_npu_whl}" \
        "${torch_npu_url}"; \
      python3 -m pip install "/tmp/${torch_npu_whl}"; \
      rm -f "/tmp/${torch_npu_whl}"; \
      python3 -c 'from importlib import metadata; assert metadata.version("torch-npu") == "'"${TORCH_NPU_VERSION}"'"; print("torch-npu", metadata.version("torch-npu"))'; \
      python3 -c 'import importlib; [importlib.import_module(name) for name in ("torchnpugen.gen_op_plugin_functions", "torchnpugen.gen_backend_stubs", "torchnpugen.struct.gen_struct_opapi")]; print("torchnpugen OK")'; \
    fi; \
    python3 -m pip uninstall -y triton || true; \
    python3 -m pip install "triton-ascend==${TRITON_ASCEND_VERSION}" --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple; \
    python3 -m pip install --target=/tmp/pybind11-target pybind11; \
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])"); \
    cp -a /tmp/pybind11-target/. "$SITE_PACKAGES/"; \
    rm -rf /tmp/pybind11-target; \
    python3 -c "import pybind11; print(pybind11.__file__)"; \
    python3 -c "from triton import Config; print('triton Config OK')"; \
    python3 -c "import site; p=site.getsitepackages()[0]+'/torchnpugen/api/cpp.py'; s=open(p).read(); old='    raise AssertionError(f\"unrecognized return type {t}\")'; new='    elif isinstance(t, OptionalType):\n        return OptionalCType(returntype_type(t.elem, mutable=False))\n    raise AssertionError(f\"unrecognized return type {t}\")'; assert old in s, 'patch target not found'; open(p,'w').write(s.replace(old,new,1)); print('torchnpugen OptionalType return patch applied')"

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
      # The offline bundle staged into release wheels lists this archive in its
      # manifest and hard-fails without it, but gtest.cmake only fetches it when
      # gtest is missing -- and the base image already has gtest.  Seed it here so
      # a wheel built from this image can carry a complete bundle.
      download "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz" \
        /opt/fla-ci/third_party/pkg/googletest-1.14.0.tar.gz; \
      clone_at "https://gitcode.com/cann/opbase.git" \
        c8d83f3e57a63a7375e89a2d6937452c0ae2e522 \
        /opt/fla-ci/third_party/opbase; \
      clone_at "https://gitcode.com/cann/catlass.git" \
        41bf90da655bba3c66d0acd7e00abe33960ecfd6 \
        /opt/fla-ci/third_party/catlass; \
    fi

WORKDIR /workspace/repo
