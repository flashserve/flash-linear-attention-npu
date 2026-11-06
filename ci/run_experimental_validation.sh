#!/usr/bin/env bash
# Compile/package validation deliberately does not claim NPU execution.
set -euo pipefail

# Disable an installed stable wheel's startup hook before any Python process.
export FLA_NPU_DISABLE_PTH=1

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
soc="${CI_SOC:-ascend950}"
ops="${CI_EXPERIMENTAL_OPS:-all}"
jobs="${CI_JOBS:-4}"
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[experimental][ERROR] Source the CANN toolkit set_env.sh first." >&2
    exit 2
fi
if [[ "${CI_EXPERIMENTAL_RUNTIME:-false}" != "false" ]]; then
    echo "[experimental][ERROR] This entry validates compilation and packaging only. Run operator accuracy tests separately." >&2
    exit 2
fi

python3 -m unittest discover -s ci/tests -p 'test_experimental_*.py' -q
python3 -m unittest discover -s experimental/_python/tests -q
bash build.sh --pkg --experimental --soc="$soc" --ops="$ops" -j"$jobs"
run_package="${repo_dir}/build_out/fla_npu_experimental_linux-$(uname -m).run"
if [[ ! -f "$run_package" ]]; then
    echo "[experimental][ERROR] Experimental run package not found: $run_package" >&2
    exit 1
fi
tmp_root="$(cd -- "${CI_TMPDIR:-/tmp}" && pwd -P)"
install_root="$(mktemp -d "${tmp_root%/}/fla-experimental-opp.XXXXXX")"
"$run_package" --install-path="$install_root" --quiet
python3 scripts/experimental_manifest.py audit \
    --install-root "$install_root" --soc "$soc" \
    --manifest build/experimental-build-manifest.json \
    | tee build_out/experimental-package-audit.json
wheel_root="$(mktemp -d "${tmp_root%/}/fla-experimental-wheel.XXXXXX")"
python3 experimental/_framework/build.py wheel \
    --manifest build/experimental-build-manifest.json \
    --install-root "$install_root" --output "$wheel_root"
python_root="$(mktemp -d "${tmp_root%/}/fla-experimental-python.XXXXXX")"
python3 -m pip install --no-deps --target "$python_root" "$wheel_root"/*.whl
source "$python_root/fla_npu_experimental/opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash"
PYTHONPATH="$python_root" python3 ci/experimental_load_smoke.py \
    | tee build_out/experimental-load-audit.json
echo "[experimental] PASS: compilation, isolated installation, ABI, wheel, and host-only loading"
echo "[experimental] NPU execution, accuracy, and A2 regression: NOT RUN"
echo "[experimental] Isolated OPP: $install_root"
echo "[experimental] Development wheel: $wheel_root"
