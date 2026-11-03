#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
device="${1:-0}"
mkdir -p build/dsl
export CATLASS_DSL_CACHE_DIR="${CATLASS_DSL_CACHE_DIR:-$PWD/build/dsl/cache}"
export CATLASS_DSL_KEEP="${CATLASS_DSL_KEEP:-all}"
export CATLASS_DSL_DUMP_DIR="${CATLASS_DSL_DUMP_DIR:-$PWD/build/dsl/dump}"
python fixtures.py generate build/fixtures
python -m catlass.bc_compile
timeout --kill-after=5s 60s python -u dsl/matmul.py build/fixtures --device "$device" --artifacts build/dsl 2>&1 | tee build/dsl-run.log
python fixtures.py check build/fixtures --backend dsl
