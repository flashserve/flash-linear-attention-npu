#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# In-container CI script for ascend950 compile-only verification (CANN 9.0.0)
#
# This script runs inside the fla-npu-ci:9.0.0-950 Docker container.
# It sources the CANN 9.0.0 environment, prepares the build cache, compiles
# the custom OPP package for ascend950, and verifies that the build
# produced a valid .run package.
# -----------------------------------------------------------------------------
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

# ---------------------------------------------------------------------------
# Select a writable TMPDIR with sufficient free space.
# Reuses the same logic as the 8.5.0 CI run_checks.sh.
# ---------------------------------------------------------------------------
select_ci_tmpdir() {
    local min_free_kb="${CI_TMPDIR_MIN_KB:-65536}"
    local candidates=()

    if [[ -n "${CI_TMPDIR:-}" ]]; then
        candidates=("$CI_TMPDIR")
    elif [[ -n "${CI_TMPDIR_CANDIDATES:-}" ]]; then
        local old_ifs="$IFS"
        IFS=':'
        read -r -a candidates <<< "$CI_TMPDIR_CANDIDATES"
        IFS="$old_ifs"
    else
        [[ -n "${TMPDIR:-}" ]] && candidates+=("$TMPDIR")
        candidates+=("$repo_dir/.ci-tmp")

        local root
        for root in /workspace /mnt /home /tmp /var/tmp; do
            [[ -d "$root" ]] && candidates+=("$root/fla-npu-ci-tmp")
        done

        local mount
        for mount in /mnt/*; do
            [[ -d "$mount" ]] && candidates+=("$mount/fla-npu-ci-tmp")
        done
    fi

    local best_dir=""
    local best_free_kb=-1
    local candidate
    local -A seen=()
    local checked=()

    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if [[ -n "${seen[$candidate]:-}" ]]; then
            continue
        fi
        seen["$candidate"]=1

        if ! mkdir -p "$candidate" 2>/dev/null; then
            checked+=("$candidate=create-failed")
            continue
        fi
        if [[ ! -w "$candidate" ]]; then
            checked+=("$candidate=not-writable")
            continue
        fi

        local free_kb
        free_kb="$(df -Pk "$candidate" 2>/dev/null | awk 'NR == 2 { print $4 }')"
        if [[ ! "$free_kb" =~ ^[0-9]+$ ]]; then
            checked+=("$candidate=df-failed")
            continue
        fi
        checked+=("$candidate=${free_kb}KB")
        if (( free_kb > best_free_kb )); then
            best_dir="$candidate"
            best_free_kb="$free_kb"
        fi
    done

    if [[ -z "$best_dir" ]]; then
        echo "[CI-950][ERROR] No writable TMPDIR candidate found. Checked: ${checked[*]:-<none>}" >&2
        return 1
    fi
    if (( best_free_kb < min_free_kb )); then
        echo "[CI-950][ERROR] No TMPDIR candidate has at least ${min_free_kb} KB free. Checked: ${checked[*]}" >&2
        return 1
    fi

    echo "$best_dir"
}

ci_tmpdir="$(select_ci_tmpdir)"
export TMPDIR="$ci_tmpdir"
echo "[CI-950] TMPDIR=$TMPDIR ($(df -h "$TMPDIR" | awk 'NR == 2 { print $4 " free" }'))"

# ---------------------------------------------------------------------------
# Clean up old CI logs (reuse shared cleanup script)
# ---------------------------------------------------------------------------
bash ci/cleanup_ci_logs.sh

# ---------------------------------------------------------------------------
# Source CANN 9.0.0 environment.
# The Docker ENTRYPOINT already sources these, but we source them again
# for safety and to ensure variables are available in all code paths.
# ---------------------------------------------------------------------------
set +u

# 1. CANN toolkit environment
if [[ -f /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash ]]; then
    source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash
    echo "[CI-950] Sourced CANN toolkit env: /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash"
fi

# 2. AscendNPU-IR (Bisheng compiler) environment
if [[ -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh ]]; then
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
    echo "[CI-950] Sourced AscendNPU-IR env: /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh"
fi

# 3. NNAL (ATB) environment
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
    echo "[CI-950] Sourced NNAL ATB env: /usr/local/Ascend/nnal/atb/set_env.sh"
fi

set -u

# ---------------------------------------------------------------------------
# Display NPU info if available (informational only; not required for
# compile-only verification).
# ---------------------------------------------------------------------------
if command -v npu-smi >/dev/null 2>&1; then
    bash ci/detect_npu.sh --summary || true
fi

# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
ci_ops="${CI_OPS:-}"
ci_jobs="${CI_JOBS:-$(nproc)}"
ci_cpack_jobs="${CI_CPACK_JOBS:-$ci_jobs}"

export CMAKE_BUILD_PARALLEL_LEVEL="$ci_cpack_jobs"
export MAKEFLAGS="${MAKEFLAGS:+$MAKEFLAGS }-j${ci_cpack_jobs}"
export FLA_NPU_SOC=ascend950

echo "[CI-950] Configuration: soc=ascend950 ops=${ci_ops:-<all>} jobs=$ci_jobs cpack_jobs=$ci_cpack_jobs"

# ---------------------------------------------------------------------------
# Prepare build cache and third_party dependencies.
# Reuses the shared prepare_ci_cache.sh which handles:
#   - Build input signature hashing
#   - Cache invalidation on input changes
#   - Seeding third_party from prewarmed Docker layer
# ---------------------------------------------------------------------------
bash ci/prepare_ci_cache.sh

# ---------------------------------------------------------------------------
# Build custom OPP package for ascend950 SoC (compile-only verification)
# ---------------------------------------------------------------------------
echo "[CI-950] Building custom OPP package for ascend950 (compile-only)..."
ascend950_ops_arg=()
if [[ -n "$ci_ops" ]]; then
    ascend950_ops_arg=(--ops="$ci_ops")
fi

echo "[CI-950] Delete old build artifacts..."
rm -rf build_out
rm -rf build

build_cmd=(bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu "${ascend950_ops_arg[@]}" -j"$ci_jobs")
echo "[CI-950] Build command: ${build_cmd[*]}"

if ! "${build_cmd[@]}"; then
    echo "[CI-950][ERROR] ascend950 package build failed." >&2
    echo "[CI-950][ERROR] Failure point: ${build_cmd[*]}" >&2
    echo "[CI-950][ERROR] Please review the build log above for compilation errors." >&2
    exit 1
fi

echo "[CI-950] ascend950 package build completed."

# ---------------------------------------------------------------------------
# Verify build results: check that the .run package was produced.
# ---------------------------------------------------------------------------
shopt -s nullglob
run_files=(build_out/fla-npu-*.run build/fla-npu-*.run)
shopt -u nullglob

if (( ${#run_files[@]} == 0 )); then
    echo "[CI-950][ERROR] No .run package found in build_out/ or build/ after ascend950 build." >&2
    echo "[CI-950][ERROR] Expected: build_out/fla-npu-*.run or build/fla-npu-*.run" >&2
    exit 1
fi

echo "[CI-950] Build artifacts verified:"
for f in "${run_files[@]}"; do
    echo "  - $f ($(du -h "$f" | awk '{print $1}'))"
done

echo "[CI-950] ascend950 compile-only verification PASSED."
