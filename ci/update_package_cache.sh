#!/usr/bin/env bash
# Refresh only the compatible Daily wheel; no NPU, installation or compilation.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${CI_IMAGE:-fla-npu-ci:9.1.0-910b}"
package_cache="${CI_PACKAGE_CACHE:-/home/huangjunzhe/ci/pkg-cache}"
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "[CI][ERROR] Missing preloaded image $image; deploy the matching CI image before updating packages." >&2
    exit 1
fi
mkdir -p "${TMPDIR:-/tmp}"
if [[ ! -d "$package_cache" ]]; then
    mkdir -p "$package_cache"
    chmod 2775 "$package_cache"
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/fla-package-update.XXXXXX")"
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -s "$work/cid" ]]; then
        docker rm -f "$(cat "$work/cid")" >/dev/null 2>&1 || true
    fi
    rm -rf -- "$work"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
proxy_env_args=()
for name in http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY; do
    if [[ -v "$name" ]]; then
        export "$name"
        proxy_env_args+=(-e "$name")
    fi
done
docker run --rm --cidfile "$work/cid" --network host \
    -v "$repo_dir:/workspace/repo:ro" -w /workspace/repo \
    -v "$package_cache:/ci-pkg-cache" -v "$work:/ci-update" \
    "${proxy_env_args[@]}" \
    -e CI_BUILD_KIND=base -e CI_SOC="${CI_SOC:-ascend910b}" \
    -e CI_BASE_REF="${CI_BASE_REF:-main}" \
    -e CI_BASE_WHEEL_INDEX_URL="${CI_BASE_WHEEL_INDEX_URL:-}" \
    -e CI_BASE_WHEEL_VERSION="${CI_BASE_WHEEL_VERSION:-}" \
    -e CI_PACKAGE_CACHE=/ci-pkg-cache -e CI_PACKAGE_INFO_FILE=/ci-update/package.json \
    "$image" python3 ci/prepare_ci_package.py --update-package &
wait "$!"
cat "$work/package.json"
