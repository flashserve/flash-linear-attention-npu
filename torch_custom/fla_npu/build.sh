#!/bin/bash
set -e
cd "$(dirname "$0")"
bash gen.sh npu_custom.yaml
rm -rf build dist flash_linear_attention_npu.egg-info fla_npu.egg-info
python3 setup.py bdist_wheel
shopt -s nullglob
wheels=(dist/flash_linear_attention_npu-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "[ERROR] Expected exactly one flash-linear-attention-npu wheel, found ${#wheels[@]}." >&2
    exit 1
fi
python3 -m pip install "${wheels[0]}" --force-reinstall --no-deps --no-cache-dir
