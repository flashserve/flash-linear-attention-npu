#!/bin/bash
set -e
cd "$(dirname "$0")"
PYTHONPATH="../../scripts${PYTHONPATH:+:$PYTHONPATH}" python3 -c 'from pathlib import Path; from stable_build_policy import validate_build_options, reject_legacy_artifacts; validate_build_options(); reject_legacy_artifacts(Path("fla_npu"))'
rm -rf build dist flash_linear_attention_npu*.egg-info fla_npu.egg-info
python3 setup.py bdist_wheel
shopt -s nullglob
wheels=(dist/flash_linear_attention_npu*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "[ERROR] Expected exactly one flash-linear-attention-npu wheel, found ${#wheels[@]}." >&2
    exit 1
fi
python3 -m pip install "${wheels[0]}" --force-reinstall --no-deps --no-cache-dir

# Install compiled OPP as a managed layer over the standalone adapter wheel.
# The base wheel and its RECORD remain unchanged.
run_pkg=""
shopt -s nullglob
for cand in ../../build_out/fla_npu_linux-*.run ../../build/fla_npu_linux-*.run; do
    if [ -n "$cand" ] && [ -s "$cand" ]; then
        run_pkg="$cand"
        break
    fi
done
shopt -u nullglob
if [ -z "$run_pkg" ]; then
    echo "[ERROR] No fla_npu_linux-*.run package found to overlay the embedded OPP into the installed wheel." >&2
    exit 1
fi
chmod +x "$run_pkg"
"$run_pkg" --install --quiet
echo ""
echo "[fla-npu] OPP submitted to the overlay manager; see activation state above. Restart Python processes."
echo "[fla-npu] The default .pth configures selected OPP paths before CANN initialization."
echo "[fla-npu] Inspect selection: python3 -m fla_npu_opp_manage --status"
