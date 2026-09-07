#!/bin/bash
set -e
cd "$(dirname "$0")"

# Same interpreter resolution as gen.sh: the caller may pass FLA_NPU_PYTHON or
# PYTHON, otherwise fall back to whatever 'python3' resolves to via PATH.
PY="${FLA_NPU_PYTHON:-${PYTHON:-python3}}"

FLA_NPU_PYTHON="$PY" bash gen.sh npu_custom.yaml
"$PY" setup.py bdist_wheel
"$PY" -m pip install ./dist/fla_npu-1.0.0-*.whl --force-reinstall --no-deps
