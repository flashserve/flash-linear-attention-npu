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

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Build the project
echo "Building the project..."
python3 setup.py clean
python3 -m build --wheel --no-isolation
python3 -m pip install dist/*.whl --force-reinstall --no-deps

# Run tests
echo "Running tests..."
for dir in tests/*/; do
    test_dir=$(basename "$dir")
    pytest "tests/$test_dir" -v
done
echo "execute samples success"
