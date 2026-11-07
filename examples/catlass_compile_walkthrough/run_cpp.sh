#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${CATLASS_ROOT:?Set CATLASS_ROOT to the CATLASS checkout}"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
device="${1:-0}"
# Pass the selected host standard library to Bisheng AND its internal linker
# subprocesses. Some systems have a newer GCC runtime without matching headers.
host_cxx="${CXX:-g++}"
host_includes="$("$host_cxx" -E -x c++ -v /dev/null 2>&1 |
    sed -n 's/^ \(\/.*\/c++\/.*\)$/\1/p' | paste -sd :)"
if [[ -n "$host_includes" ]]; then
    export CPLUS_INCLUDE_PATH="$host_includes${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
host_library="$("$host_cxx" -print-file-name=libstdc++.so)"
if [[ -f "$host_library" ]]; then
    export LIBRARY_PATH="$(dirname "$host_library")${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
python fixtures.py generate build/fixtures
cmake -S cpp -B build/cpp -DCATLASS_ROOT="$CATLASS_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel 2 --verbose 2>&1 | tee build/cpp-build.log
for case_name in zero identity pattern; do
    timeout 60s build/cpp/matmul_cpp "build/fixtures/$case_name" "$device"
done
python fixtures.py check build/fixtures --backend cpp
