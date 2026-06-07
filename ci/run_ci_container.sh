#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

image="${CI_IMAGE:-fla-npu-ci:8.5.0-910b}"
container_name="${CI_CONTAINER_NAME:-fla-npu-ci-$(date +%s)}"
cache_root="${CI_CACHE_ROOT:-}"
npu_lock_fd=""
npu_lock_file=""

if [[ -z "$cache_root" ]]; then
    if [[ -d /workspace ]]; then
        cache_root="/workspace/flash-linear-attention-npu-ci/cache"
    else
        cache_root="$repo_dir/.ci-cache"
    fi
fi

release_npu_lock() {
    if [[ -n "${npu_lock_fd:-}" ]]; then
        flock -u "$npu_lock_fd" >/dev/null 2>&1 || true
        eval "exec ${npu_lock_fd}>&-" || true
        echo "[CI] Released NPU lock: ${npu_lock_file}"
        npu_lock_fd=""
        npu_lock_file=""
    fi
}

acquire_npu_lock() {
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "[CI][ERROR] npu-smi is not available on host." >&2
        exit 1
    fi
    if ! command -v flock >/dev/null 2>&1; then
        echo "[CI][ERROR] flock is required for NPU locking." >&2
        exit 1
    fi

    local lock_dir="${CI_NPU_LOCK_DIR:-/tmp}"
    local lock_wait_seconds="${CI_NPU_LOCK_WAIT_SECONDS:-14400}"
    local lock_retry_seconds="${CI_NPU_LOCK_RETRY_SECONDS:-10}"
    local started_at="$SECONDS"
    mkdir -p "$lock_dir"

    while true; do
        local candidates=()
        mapfile -t candidates < <(bash ci/detect_npu.sh --candidates)
        if (( ${#candidates[@]} == 0 )); then
            echo "[CI][ERROR] No NPU device was found on host." >&2
            exit 1
        fi

        local id
        for id in "${candidates[@]}"; do
            local candidate_lock="${lock_dir}/fla-npu-ci-npu-${id}.lock"
            local fd
            exec {fd}>"$candidate_lock"
            if flock -n "$fd"; then
                npu_lock_fd="$fd"
                npu_lock_file="$candidate_lock"
                eval "$(bash ci/detect_npu.sh --env-for "$id")"
                trap release_npu_lock EXIT
                trap 'release_npu_lock; exit 130' INT
                trap 'release_npu_lock; exit 143' TERM
                echo "[CI] Acquired NPU lock: $npu_lock_file"
                return
            fi
            eval "exec ${fd}>&-"
        done

        local elapsed=$((SECONDS - started_at))
        if [[ "$lock_wait_seconds" != "0" && "$elapsed" -ge "$lock_wait_seconds" ]]; then
            echo "[CI][ERROR] Timed out waiting for an unlocked NPU after ${lock_wait_seconds}s." >&2
            exit 1
        fi
        echo "[CI] All detected NPU devices are locked; retrying in ${lock_retry_seconds}s."
        sleep "$lock_retry_seconds"
    done
}

if ! docker image inspect "$image" >/dev/null 2>&1 || [[ "${CI_REBUILD_IMAGE:-false}" == "true" ]]; then
    echo "[CI] Building Docker image: $image"
    docker build -t "$image" -f ci/Dockerfile .
fi

acquire_npu_lock

if [[ "${CI_REQUIRE_HEALTHY_NPU:-false}" == "true" && "${NPU_SELECTED_HEALTH:-}" != "OK" ]]; then
    echo "[CI][ERROR] Selected NPU ${NPU_SELECTED_DEVICE:-unknown} health is ${NPU_SELECTED_HEALTH:-unknown}." >&2
    exit 1
fi

device_args=()
for dev in /dev/davinci[0-9]* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    if [[ -e "$dev" ]]; then
        device_args+=(--device "$dev")
    fi
done
if [[ "${CI_DOCKER_PRIVILEGED:-true}" == "true" ]]; then
    device_args=(--privileged "${device_args[@]}")
fi

third_party_cache="${CI_THIRD_PARTY_CACHE:-$cache_root/third_party}"
mkdir -p "$third_party_cache"

mount_args=(
    -v "$repo_dir:/workspace/repo"
    -v "$third_party_cache:/workspace/repo/third_party"
    -w /workspace/repo
)
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

echo "[CI] Running $container_name on NPU ${NPU_SELECTED_DEVICE} (${NPU_SELECTED_NAME}, health=${NPU_SELECTED_HEALTH}, free=${NPU_SELECTED_FREE})"
echo "[CI] third_party cache: $third_party_cache"

docker run --rm \
    --name "$container_name" \
    --network host \
    --ipc host \
    "${device_args[@]}" \
    "${mount_args[@]}" \
    -e ASCEND_RT_VISIBLE_DEVICES="${NPU_SELECTED_DEVICE}" \
    -e NPU_SELECTED_DEVICE="${NPU_SELECTED_DEVICE}" \
    -e NPU_SELECTED_NAME="${NPU_SELECTED_NAME}" \
    -e NPU_SELECTED_HEALTH="${NPU_SELECTED_HEALTH}" \
    -e NPU_SELECTED_FREE="${NPU_SELECTED_FREE}" \
    -e NPU_SOC="${NPU_SOC}" \
    -e CI_CONTAINER_DEVICE="${CI_CONTAINER_DEVICE:-0}" \
    -e CI_MODE="${CI_MODE:-quick}" \
    -e CI_SOC="${CI_SOC:-${NPU_SOC}}" \
    -e CI_OPS="${CI_OPS:-}" \
    -e CI_JOBS="${CI_JOBS:-}" \
    -e CI_CPACK_JOBS="${CI_CPACK_JOBS:-}" \
    -e CI_FORCE_CLEAN_CACHE="${CI_FORCE_CLEAN_CACHE:-false}" \
    -e CI_LOG_CLEANUP_ENABLED="${CI_LOG_CLEANUP_ENABLED:-true}" \
    -e CI_LOG_RETENTION_DAYS="${CI_LOG_RETENTION_DAYS:-7}" \
    -e CI_LOG_CLEANUP_DIRS="${CI_LOG_CLEANUP_DIRS:-}" \
    -e CI_SEED_THIRD_PARTY="${CI_SEED_THIRD_PARTY:-true}" \
    -e CI_BUILD_TORCH_CUSTOM="${CI_BUILD_TORCH_CUSTOM:-false}" \
    -e CI_RUN_TORCH_TESTS="${CI_RUN_TORCH_TESTS:-false}" \
    -e CI_RUN_EXAMPLE_ST="${CI_RUN_EXAMPLE_ST:-true}" \
    -e CI_EXAMPLE_CASES_FILE="${CI_EXAMPLE_CASES_FILE:-ci/example_st_cases.json}" \
    -e CI_EXAMPLE_CASE_FILTER="${CI_EXAMPLE_CASE_FILTER:-}" \
    -e CI_TEST_OP="${CI_TEST_OP:-}" \
    "$image" \
    bash ci/run_checks.sh
