#!/bin/bash
set -e
cd "$(dirname "$0")"
PYTHONPATH="../../scripts${PYTHONPATH:+:$PYTHONPATH}" python3 -c 'from pathlib import Path; from stable_build_policy import validate_build_options, reject_legacy_artifacts; validate_build_options(); reject_legacy_artifacts(Path("fla_npu"))'
rm -rf build dist flash_linear_attention_npu*.egg-info fla_npu.egg-info
run_pkg=""
shopt -s nullglob
for cand in ../../build_out/fla_npu_linux-*.run ../../build/fla_npu_linux-*.run; do
    if [ -n "$cand" ] && [ -s "$cand" ]; then
        run_pkg="$(realpath "$cand")"
        break
    fi
done
shopt -u nullglob
if [ -z "$run_pkg" ]; then
    echo "[ERROR] No fla_npu_linux-*.run package found to embed in the base wheel." >&2
    exit 1
fi
# Use the root wheel builder so the prebuilt OPP is validated and included in the wheel RECORD.
adapter_dir="$PWD"
(cd ../.. && python3 setup.py build --build-base "$adapter_dir/build" \
    build_py --opp-run "$run_pkg" bdist_wheel --dist-dir "$adapter_dir/dist")
shopt -s nullglob
wheels=(dist/flash_linear_attention_npu*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "[ERROR] Expected exactly one flash-linear-attention-npu wheel, found ${#wheels[@]}." >&2
    exit 1
fi
python3 -m pip install "${wheels[0]}" --force-reinstall --no-deps --no-cache-dir
