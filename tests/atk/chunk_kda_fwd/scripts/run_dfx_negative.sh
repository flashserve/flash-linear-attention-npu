#!/bin/bash
# 构建并运行 chunk_kda_fwd 的 ACLNN 负向用例（host 侧，不执行 kernel）。
# 依赖：已编出的 fla_npu_*.run、CANN 9.2 环境。
#   FLA_NPU_RUN=/path/to/fla_npu_linux-x86_64.run ./run_dfx_negative.sh
set -euo pipefail

CANN=${CANN_HOME:-/usr/local/Ascend/cann-9.2.0-beta.2}
RUN=${FLA_NPU_RUN:?set FLA_NPU_RUN to the built fla_npu_*.run}
WORK=${WORKDIR:-$(mktemp -d)}
HERE=$(cd "$(dirname "$0")" && pwd)

bash "$RUN" --noexec --extract="$WORK" >/dev/null
V=$(find "$WORK" -maxdepth 5 -type d -name op_api | head -1)
if [ -z "$V" ]; then
    echo "op_api not found in $WORK" >&2
    exit 2
fi

g++ -std=c++17 -O1 -o "$WORK/probe_neg" "$HERE/dfx_negative_probe.cpp" \
    -I"$V/include/aclnnop" -I"$CANN/include" -I"$CANN/x86_64-linux/include" \
    -I"$CANN/x86_64-linux/include/aclnn" -I"$CANN/x86_64-linux/include/aclnnop" \
    -L"$V/lib" -lcust_opapi \
    -L"$CANN/lib64" -lascendcl -lnnopbase -ldl \
    -Wl,--allow-shlib-undefined -Wl,-rpath-link,"$CANN/runtime/lib64"

export LD_LIBRARY_PATH="$V/lib:$CANN/lib64:$CANN/runtime/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$V/.."
timeout 120 "$WORK/probe_neg"
