#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# CI orchestration for ascend950 compile-only verification (CANN 9.0.0)
#
# This script runs AFTER the 8.5.0 (ascend910b) CI pipeline succeeds.
# It builds a dedicated Docker image from ci/9.0.0/Dockerfile and runs
# ci/9.0.0/run_checks.sh inside the container to verify that the code
# compiles for ascend950.
#
# Locking is NOT handled here: the parent ci/run_ci_container.sh holds a
# shared NPU lock for the entire 8.5.0 + 9.0.0 run and releases it once on
# exit, so this script must not acquire or release any lock of its own.
# -----------------------------------------------------------------------------
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

image="${CI_IMAGE:-fla-npu-ci:9.0.0-910b}"
container_name="${CI_CONTAINER_NAME:-fla-npu-ci-910b-$(date +%s)}"
dockerfile="ci/9.0.0/Dockerfile"
cache_root="${CI_CACHE_ROOT:-}"

if [[ -z "$cache_root" ]]; then
    if [[ -d /workspace ]]; then
        cache_root="/workspace/flash-linear-attention-npu-ci/cache"
    else
        cache_root="$repo_dir/.ci-cache"
    fi
fi

# ---------------------------------------------------------------------------
# Build Docker image if it does not exist or rebuild is requested.
# ---------------------------------------------------------------------------
if ! docker image inspect "$image" >/dev/null 2>&1 || [[ "${CI_REBUILD_IMAGE:-false}" == "true" ]]; then
    echo "[CI-950] Building Docker image: $image (from $dockerfile)"
    docker build -t "$image" -f "$dockerfile" .
fi

# ---------------------------------------------------------------------------
# Device passthrough: mount NPU devices if available (not strictly required
# for compile-only, but kept for consistency and potential future use).
# ---------------------------------------------------------------------------
device_args=()
if [[ "${CI_DOCKER_PRIVILEGED:-false}" == "true" ]]; then
    device_args=(--privileged)
else
    for dev in /dev/davinci[0-9]* /dev/davinciManager /dev/devmm_svm /dev/hisi_hdc; do
        if [[ -e "$dev" ]]; then
            device_args+=(--device "$dev")
        fi
    done
fi

# ---------------------------------------------------------------------------
# Cache directories: third_party packages are shared between host and
# container to avoid re-downloading on every run.
# ---------------------------------------------------------------------------
third_party_cache="${CI_THIRD_PARTY_CACHE:-$cache_root/third_party}"
mkdir -p "$third_party_cache"

mount_args=(
    -v "$repo_dir:/workspace/repo"
    -v "$third_party_cache:/workspace/repo/third_party"
    -w /workspace/repo
)

# Mount NPU driver files if available (for potential on-device checks)
for path in \
    /usr/local/dcmi \
    /usr/local/bin/npu-smi \
    /usr/local/Ascend/driver/lib64 \
    /usr/local/Ascend/driver/version.info \
    /etc/ascend_install.info; do
    if [[ -e "$path" ]]; then
        mount_args+=(-v "$path:$path")
    fi
done

echo "[CI-950] Running $container_name (ascend950 compile-only verification)"
echo "[CI-950] third_party cache: $third_party_cache"
echo "[CI-950] container TMPDIR: ${CI_TMPDIR:-auto}"

container_command=(bash ci/9.0.0/run_checks.sh)
if [[ -n "${CI_CONTAINER_COMMAND:-}" ]]; then
    container_command=(bash -lc "$CI_CONTAINER_COMMAND")
fi

docker run --rm \
    --name "$container_name" \
    --network host \
    --ipc host \
    "${device_args[@]}" \
    "${mount_args[@]}" \
    -e CI_MODE="${CI_MODE:-quick}" \
    -e CI_SOC=ascend950 \
    -e CI_OPS="${CI_OPS:-}" \
    -e CI_JOBS="${CI_JOBS:-}" \
    -e CI_CPACK_JOBS="${CI_CPACK_JOBS:-}" \
    -e CI_FORCE_CLEAN_CACHE="${CI_FORCE_CLEAN_CACHE:-false}" \
    -e CI_LOG_CLEANUP_ENABLED="${CI_LOG_CLEANUP_ENABLED:-true}" \
    -e CI_LOG_RETENTION_DAYS="${CI_LOG_RETENTION_DAYS:-7}" \
    -e CI_LOG_CLEANUP_DIRS="${CI_LOG_CLEANUP_DIRS:-}" \
    -e CI_SEED_THIRD_PARTY="${CI_SEED_THIRD_PARTY:-true}" \
    -e CI_TMPDIR="${CI_TMPDIR:-}" \
    -e CI_TMPDIR_CANDIDATES="${CI_TMPDIR_CANDIDATES:-}" \
    -e CI_TMPDIR_MIN_KB="${CI_TMPDIR_MIN_KB:-}" \
    -e FLA_NPU_SOC=ascend950 \
    -e FLA_NPU_INCREMENTAL_BUILD="${FLA_NPU_INCREMENTAL_BUILD:-false}" \
    -e FLA_NPU_LOCAL_VERSION="${FLA_NPU_LOCAL_VERSION:-}" \
    "$image" \
    "${container_command[@]}"

echo "[CI-950] ascend950 compile-only verification completed successfully."
