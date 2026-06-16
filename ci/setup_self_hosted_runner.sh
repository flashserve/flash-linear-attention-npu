#!/usr/bin/env bash
set -euo pipefail

repo_url=""
runner_token=""
runner_root="${RUNNER_ROOT:-/workspace/actions-runner/flash-linear-attention-npu}"
runner_name="${RUNNER_NAME:-$(hostname)-flash-linear-attention-npu}"
runner_labels="${RUNNER_LABELS:-linux,arm64,npu,flash-linear-attention-npu}"
runner_version="${RUNNER_VERSION:-latest}"

usage() {
    cat <<EOF
Usage: $0 --url <github-repo-url> --token <registration-token>

Required:
  --url      GitHub repository URL, for example https://github.com/flashserve/flash-linear-attention-npu
  --token    Self-hosted runner registration token generated from repository settings

Optional env:
  RUNNER_ROOT       Default: /workspace/actions-runner/flash-linear-attention-npu
  RUNNER_NAME       Default: <hostname>-flash-linear-attention-npu
  RUNNER_LABELS     Default: linux,arm64,npu,flash-linear-attention-npu
  RUNNER_VERSION    Default: latest
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)
            repo_url="$2"
            shift 2
            ;;
        --token)
            runner_token="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$repo_url" || -z "$runner_token" ]]; then
    usage >&2
    exit 2
fi

arch="$(uname -m)"
case "$arch" in
    aarch64|arm64) runner_arch="arm64" ;;
    x86_64|amd64) runner_arch="x64" ;;
    *)
        echo "Unsupported runner arch: $arch" >&2
        exit 1
        ;;
esac

if [[ "$runner_version" == "latest" ]]; then
    runner_version="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
        | sed -n 's/.*"tag_name": "v\([^"]*\)".*/\1/p' \
        | head -1)"
fi
if [[ -z "$runner_version" ]]; then
    echo "Failed to resolve GitHub Actions runner version." >&2
    exit 1
fi

mkdir -p "$runner_root"
cd "$runner_root"

archive="actions-runner-linux-${runner_arch}-${runner_version}.tar.gz"
url="https://github.com/actions/runner/releases/download/v${runner_version}/${archive}"

if [[ ! -f ./config.sh ]]; then
    echo "[runner] Downloading $url"
    curl -fL "$url" -o "$archive"
    tar xzf "$archive"
fi

./config.sh \
    --url "$repo_url" \
    --token "$runner_token" \
    --name "$runner_name" \
    --labels "$runner_labels" \
    --work _work \
    --unattended \
    --replace

if [[ "$(id -u)" == "0" && -f ./svc.sh ]]; then
    ./svc.sh install || true
    ./svc.sh start
else
    echo "[runner] Start the runner with: cd $runner_root && ./run.sh"
fi
