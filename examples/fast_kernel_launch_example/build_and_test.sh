#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e
cd "$(dirname "$0")"

OP_NAME="${1:-}"

usage() {
    echo "Usage: $0 [OP_NAME]"
    echo "  OP_NAME: operator name to test (e.g., add, grouped_matmul, chunk_bwd_dv_local, chunk_fwd_o)"
    echo "  If not specified, all tests will be run."
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

if [[ -n "$OP_NAME" && ! -d "tests/$OP_NAME" ]]; then
    echo "Error: test directory 'tests/$OP_NAME' not found."
    echo "Available operators:"
    for dir in tests/*/; do
        echo "  $(basename "$dir")"
    done
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
if [[ -n "$OP_NAME" ]]; then
    pytest "tests/$OP_NAME" -v
else
    for dir in tests/*/; do
        test_dir=$(basename "$dir")
        pytest "tests/$test_dir" -v
    done
fi
echo "execute samples success"
