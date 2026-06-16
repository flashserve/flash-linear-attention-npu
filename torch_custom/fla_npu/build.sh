#!/bin/bash
set -e
cd "$(dirname "$0")"
bash gen.sh npu_custom.yaml
python3 setup.py bdist_wheel
pip3 install ./dist/fla_npu-1.0.0-*.whl --force-reinstall --no-deps
