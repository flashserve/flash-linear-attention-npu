#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

source_dir="$(realpath "$SCRIPT_DIR/..")"
repo_url=""
ref=""
soc="${FLA_NPU_SOC:-}"
device="0"
work_root="${PWD}/outputs/kda-main-benchmark"
cann_env=""
conda_init=""
conda_env=""
case_filter="all"
warm_up="5"
launch_count="1"
case_timeout="900"
decode_step="1"
clone_retries="3"
aic_metrics="Default"
ops="chunk_kda_fwd,kda_gate_cumsum,chunk_gated_delta_rule_fwd_h"

usage() {
    cat <<'EOF'
Usage: bash scripts/benchmark_kda_main.sh [options]

Build the current checkout into an isolated fla_npu wheel, then profile the KDA
A5 dense matrix (case IDs 250-257) with msopprof. Results are written
as CSV, Markdown, and JSON. Remote
fetching is opt-in; pass both --repo-url and --ref when it is required.

Options:
  --work-root DIR        Parent directory for source, wheel, profiles, and logs
  --source-dir DIR       Existing source checkout (default: this repository)
  --repo-url URL         Repository URL for an explicit remote fetch
  --ref REF              Branch, tag, SHA, or refs/pull/<N>/head to fetch
  --soc SOC              Required: ascend910b, ascend910_93, or ascend950
  --device ID            Physical NPU exposed to the benchmark (default: 0)
  --cann-env FILE        CANN set_env.sh to source before building
  --conda-init FILE      Conda profile script; required with --conda-env
  --conda-env NAME       Conda environment to activate
  --cases IDS            Comma-separated dense case IDs/keys, or all (default: all)
  --warm-up N            msopprof replay warm-up count (default: 5)
  --launch-count N       Maximum matching kernels to collect (default: 1)
  --case-timeout SEC     Timeout for one worker/profile command (default: 900)
  --decode-step N        Deprecated compatibility option; only 1 is accepted
  --aic-metrics NAME     msopprof AI Core metrics set (default: Default)
                         Keep Default for full MTE/Cube/Vector/Fixpipe details
  --ops IDS              Comma-separated operators for the scoped wheel build
  --clone-retries N      Source fetch attempts (default: 3)
  -h, --help             Show this help

Environment:
  FLA_NPU_SOC            SOC used when --soc is omitted
  FLA_NPU_ALLOWED_ROOT   Optional absolute root. The script refuses writes
                         outside it when set.

Examples:
  bash scripts/benchmark_kda_main.sh --soc ascend910b \
    --work-root "$PWD/outputs/kda-perf"
  bash scripts/benchmark_kda_main.sh --soc ascend950 \
    --cann-env /path/to/Ascend/ascend-toolkit/set_env.sh
  bash scripts/benchmark_kda_main.sh --soc ascend950 \
    --repo-url https://github.com/example/project.git \
    --ref refs/pull/276/head
EOF
}

source_env_file() {
    local path="$1"
    set +u
    # shellcheck disable=SC1090
    source "$path"
    set -u
}

cmake_version_is_supported() {
    local version="$1"
    if [[ "$version" =~ ^([0-9]+)\.([0-9]+)(\.[0-9]+)?$ ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"
        ((major == 3 && minor >= 16))
        return
    fi
    return 1
}

while (($#)); do
    case "$1" in
        --work-root) work_root="$2"; shift 2 ;;
        --source-dir) source_dir="$2"; shift 2 ;;
        --repo-url) repo_url="$2"; shift 2 ;;
        --ref) ref="$2"; shift 2 ;;
        --soc) soc="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --cann-env) cann_env="$2"; shift 2 ;;
        --conda-init) conda_init="$2"; shift 2 ;;
        --conda-env) conda_env="$2"; shift 2 ;;
        --cases) case_filter="$2"; shift 2 ;;
        --warm-up) warm_up="$2"; shift 2 ;;
        --launch-count) launch_count="$2"; shift 2 ;;
        --case-timeout) case_timeout="$2"; shift 2 ;;
        --decode-step) decode_step="$2"; shift 2 ;;
        --aic-metrics) aic_metrics="$2"; shift 2 ;;
        --aic-metrics=*) aic_metrics="${1#*=}"; shift ;;
        --ops) ops="$2"; shift 2 ;;
        --clone-retries) clone_retries="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$soc" ]] || { echo "--soc is required (or set FLA_NPU_SOC)" >&2; exit 2; }
case "$soc" in
    ascend910b|ascend910_93|ascend950) ;;
    *) echo "Unsupported SOC: $soc" >&2; exit 2 ;;
esac
[[ "$device" =~ ^[0-9]+$ ]] || { echo "--device must be a non-negative integer" >&2; exit 2; }
[[ "$warm_up" =~ ^[0-9]+$ ]] || { echo "--warm-up must be a non-negative integer" >&2; exit 2; }
[[ "$launch_count" =~ ^[1-9][0-9]*$ ]] || { echo "--launch-count must be positive" >&2; exit 2; }
[[ "$case_timeout" =~ ^[1-9][0-9]*$ ]] || { echo "--case-timeout must be positive" >&2; exit 2; }
[[ "$decode_step" == "1" ]] || { echo "--decode-step is deprecated and only accepts 1" >&2; exit 2; }
[[ "$clone_retries" =~ ^[1-9][0-9]*$ ]] || { echo "--clone-retries must be positive" >&2; exit 2; }
[[ -n "$aic_metrics" ]] || { echo "--aic-metrics must not be empty" >&2; exit 2; }
[[ -n "$ops" ]] || { echo "--ops must not be empty" >&2; exit 2; }
if [[ -n "$repo_url" || -n "$ref" ]]; then
    [[ -n "$repo_url" && -n "$ref" ]] || {
        echo "--repo-url and --ref must be provided together" >&2
        exit 2
    }
fi

if [[ -n "$conda_env" ]]; then
    [[ -n "$conda_init" ]] || { echo "--conda-init is required with --conda-env" >&2; exit 2; }
    [[ -f "$conda_init" ]] || { echo "Conda profile script not found: $conda_init" >&2; exit 2; }
    source_env_file "$conda_init"
    conda activate "$conda_env"
fi

if [[ -n "$cann_env" ]]; then
    [[ -f "$cann_env" ]] || { echo "CANN environment script not found: $cann_env" >&2; exit 2; }
    source_env_file "$cann_env"
fi

if [[ -z "${ASCEND_HOME_PATH:-}" && -z "${ASCEND_OPP_PATH:-}" ]]; then
    for candidate in \
        /usr/local/Ascend/cann/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        if [[ -f "$candidate" ]]; then
            echo "Auto-loading CANN environment: $candidate"
            source_env_file "$candidate"
            break
        fi
    done
fi

for command_name in cksum git python3 npu-smi msopprof realpath timeout; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Required command not found: $command_name" >&2
        exit 1
    }
done
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' || {
    echo "Python 3.9 or newer is required; found $(python3 --version 2>&1)" >&2
    exit 1
}
[[ -n "${ASCEND_HOME_PATH:-}" || -n "${ASCEND_OPP_PATH:-}" ]] || {
    echo "CANN is not active. Source set_env.sh or pass --cann-env." >&2
    exit 1
}

mkdir -p -- "$work_root"
work_root="$(realpath "$work_root")"
[[ "$work_root" != "/" ]] || { echo "Refusing to use / as --work-root" >&2; exit 2; }
if [[ -n "${FLA_NPU_ALLOWED_ROOT:-}" ]]; then
    allowed_root="$(realpath "${FLA_NPU_ALLOWED_ROOT}")"
    case "$work_root/" in
        "$allowed_root"/*) ;;
        *) echo "Work root is outside FLA_NPU_ALLOWED_ROOT: $work_root" >&2; exit 2 ;;
    esac
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$work_root/run_${timestamp}_$$"
cache_root="$work_root/cache"
fetched_source_dir="$cache_root/source"
wheel_dir="$run_dir/wheel"
pip_cache_dir="$cache_root/pip"
tmp_dir="$run_dir/tmp"
result_dir="$run_dir/results"
ascend_log_dir="$run_dir/ascend_logs"
base_python="$(realpath "$(command -v python3)")"
python_key="$("$base_python" -c 'import sys; print(sys.executable, sys.prefix, sys.version_info[:2])' | cksum | awk '{print $1}')"
venv_dir="$cache_root/venv_${python_key}"
mkdir -p -- "$run_dir" "$cache_root" "$pip_cache_dir" "$wheel_dir" "$tmp_dir" "$result_dir" "$ascend_log_dir"
export ASCEND_PROCESS_LOG_PATH="$ascend_log_dir"
export ASCEND_HOST_LOG_FILE_NUM="${ASCEND_HOST_LOG_FILE_NUM:-10}"
echo "Ascend process logs: $ASCEND_PROCESS_LOG_PATH"

on_error() {
    status=$?
    echo "Benchmark failed with status $status. Logs and partial outputs: $run_dir" >&2
    exit "$status"
}
trap on_error ERR

echo "[1/6] Checking NPU $device"
npu-smi info -t board -i "$device" >/dev/null

if [[ -n "$repo_url" ]]; then
    echo "[2/6] Fetching $repo_url at $ref"
    source_dir="$fetched_source_dir"
    if [[ ! -d "$source_dir/.git" ]]; then
        git init "$source_dir"
        git -C "$source_dir" remote add origin "$repo_url"
    elif [[ "$(git -C "$source_dir" remote get-url origin)" != "$repo_url" ]]; then
        echo "Cached source uses a different origin: $source_dir" >&2
        exit 1
    fi
    fetch_succeeded="0"
    for ((attempt = 1; attempt <= clone_retries; attempt++)); do
        echo "Source fetch attempt $attempt/$clone_retries"
        fetch_args=(
            -C "$source_dir"
            -c http.lowSpeedLimit=1000
            -c http.lowSpeedTime=30
            fetch --depth 1 --filter=blob:none origin "$ref"
        )
        if GIT_TERMINAL_PROMPT=0 timeout 90 git "${fetch_args[@]}"; then
            fetch_succeeded="1"
            break
        fi
        ((attempt == clone_retries)) || sleep 5
    done
    [[ "$fetch_succeeded" == "1" ]] || { echo "Unable to fetch $repo_url at $ref" >&2; exit 1; }
    git -C "$source_dir" checkout --detach --force FETCH_HEAD
else
    source_dir="$(realpath "$source_dir")"
    [[ -d "$source_dir/.git" || -f "$source_dir/.git" ]] || {
        echo "--source-dir is not a Git checkout: $source_dir" >&2
        exit 2
    }
    echo "[2/6] Using local source checkout: $source_dir"
    if [[ -n "$(git -C "$source_dir" status --short --untracked-files=no)" ]]; then
        echo "Source checkout has tracked changes; the wheel will include them." >&2
    fi
fi
commit="$(git -C "$source_dir" rev-parse HEAD)"
runner_script="$source_dir/scripts/benchmark_kda_matrix.py"
[[ -f "$runner_script" ]] || { echo "Benchmark runner is missing: $runner_script" >&2; exit 1; }
echo "Source commit: $commit"

echo "[3/6] Creating isolated Python environment"
if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$base_python" -m venv --system-site-packages "$venv_dir"
else
    echo "Reusing cached Python environment: $venv_dir"
fi
venv_python="$venv_dir/bin/python"
export PATH="$venv_dir/bin:$PATH"
export PIP_CACHE_DIR="$pip_cache_dir"
"$venv_python" -c 'import torch, torch_npu; print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")'

echo "[4/6] Installing and checking build dependencies"
build_requirements=(
    "setuptools>=70.1"
    wheel
    packaging
    psutil
)
cmake_version=""
if command -v cmake >/dev/null 2>&1; then
    cmake_version="$(cmake --version | sed -n '1s/^cmake version //p')"
fi
if ! cmake_version_is_supported "$cmake_version"; then
    build_requirements+=("cmake>=3.16,<4")
fi
if ! command -v patch >/dev/null 2>&1; then
    build_requirements+=("patch-ng==1.19.1")
fi
"$venv_python" -m pip install "${build_requirements[@]}"

if ! command -v patch >/dev/null 2>&1; then
    export FLA_NPU_PATCH_PYTHON="$venv_python"
    patch_wrapper="$venv_dir/bin/patch"
    cat > "$patch_wrapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

: "${FLA_NPU_PATCH_PYTHON:?FLA_NPU_PATCH_PYTHON is required}"
case "${1:-}" in
    --version|-h|--help)
        exec "$FLA_NPU_PATCH_PYTHON" -m patch_ng "$@"
        ;;
    -p1)
        shift
        set -- -p0 "$@"
        ;;
esac
patch_input="$(mktemp "${TMPDIR:-/tmp}/fla-npu-patch.XXXXXX")"
trap 'rm -f -- "$patch_input"' EXIT
cat > "$patch_input"
"$FLA_NPU_PATCH_PYTHON" -m patch_ng "$@" "$patch_input"
EOF
    chmod +x "$patch_wrapper"
    echo "GNU patch not found; using the isolated patch-ng compatibility wrapper"
fi

for build_command in cmake make patch; do
    command -v "$build_command" >/dev/null 2>&1 || {
        echo "Required build command not found: $build_command" >&2
        exit 1
    }
done
echo "cmake=$(cmake --version | sed -n '1p')"
echo "patch=$(patch --version | sed -n '1p')"
export FLA_NPU_SOC="$soc"
export FLA_NPU_OPS="$ops"
export FLA_NPU_BUILD_LEGACY_EXTENSION="FALSE"
export TMPDIR="$tmp_dir"
export TORCH_EXTENSIONS_DIR="$run_dir/torch_extensions"
export ASCEND_RT_VISIBLE_DEVICES="$device"
mkdir -p -- "$TORCH_EXTENSIONS_DIR"
"$venv_python" "$source_dir/scripts/check_npu_env.py" --build-only

echo "[5/6] Building and installing the main-branch wheel"
(
    cd "$source_dir"
    "$venv_python" -m pip wheel --no-build-isolation --no-deps . -w "$wheel_dir"
)
mapfile -t wheels < <(find "$wheel_dir" -maxdepth 1 -type f -name 'flash_linear_attention_npu-*.whl' -print)
if ((${#wheels[@]} != 1)); then
    echo "Expected one fla_npu wheel, found ${#wheels[@]}" >&2
    exit 1
fi
"$venv_python" -m pip install --force-reinstall --no-deps "${wheels[0]}"
"$venv_python" "$source_dir/scripts/check_kda_benchmark_wheel.py"

echo "[6/6] Profiling the KDA forward matrix with msopprof"
runner=(
    "$runner_script"
    --output-dir "$result_dir"
    --repo-dir "$source_dir"
    --repo-commit "$commit"
    --soc "$soc"
    --device-visible-id 0
    --cases "$case_filter"
    --warm-up "$warm_up"
    --launch-count "$launch_count"
    --case-timeout "$case_timeout"
    --aic-metrics "$aic_metrics"
)
"$venv_python" "${runner[@]}"

trap - ERR
echo "Completed. Results: $result_dir/results.md"
