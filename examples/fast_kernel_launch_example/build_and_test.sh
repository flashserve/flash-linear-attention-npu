#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------
# Adapted for flash-linear-attention-npu by Tianjin University.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OP_NAME="${1:-}"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 [OP_NAME]"
    echo "  OP_NAME: operator name to test (e.g., add, grouped_matmul, chunk_bwd_dv_local, chunk_fwd_o)"
    echo "  If not specified, all tests will be run."
    exit 1
}

if [[ "$OP_NAME" == "-h" || "$OP_NAME" == "--help" ]]; then
    usage
fi

canonical_test() {
    echo "$REPO_ROOT/tests/operators/$1/routes/test_fast_kernel_$1.py"
}

if [[ -n "$OP_NAME" && ! -d "tests/$OP_NAME" && ! -f "$(canonical_test "$OP_NAME")" ]]; then
    echo "Error: no fast-kernel test is registered for '$OP_NAME'."
    echo "Available operators:"
    for dir in tests/*/; do
        echo "  $(basename "$dir")"
    done
    while IFS= read -r test_file; do
        basename "$(dirname "$(dirname "$test_file")")"
    done < <(find "$REPO_ROOT/tests/operators" -path "*/routes/test_fast_kernel_*.py" -print | sort)
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Build the project
echo "Building the project..."
if [[ -n "$OP_NAME" ]]; then
    export FAST_KERNEL_OP_NAME="$OP_NAME"
else
    unset FAST_KERNEL_OP_NAME
fi
python3 setup.py clean
python3 setup.py bdist_wheel
python3 -m pip install dist/*.whl --force-reinstall --no-deps

# Run tests
echo "Running tests..."
export FLA_NPU_RUN_FAST_KERNEL_TESTS=1
if [[ -n "$OP_NAME" ]]; then
    canonical="$(canonical_test "$OP_NAME")"
    if [[ -f "$canonical" ]]; then
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" pytest "$canonical" -v
    else
        pytest "tests/$OP_NAME" -v
    fi
else
    for dir in tests/*/; do
        test_dir=$(basename "$dir")
        pytest "tests/$test_dir" -v
    done
    while IFS= read -r test_file; do
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" pytest "$test_file" -v
    done < <(find "$REPO_ROOT/tests/operators" -path "*/routes/test_fast_kernel_*.py" -print | sort)
fi
echo "execute samples success"
