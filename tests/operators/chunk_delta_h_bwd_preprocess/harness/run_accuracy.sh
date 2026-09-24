#!/usr/bin/env bash
# 在装有本算子自定义 OPP 的机器上：生成用例 → 编译并运行 aclnn 取数程序 → 与 CPU 标杆比对
#
#   INSTALL=<custom opp 安装目录> CASE_DIR=<用例目录> bash run_accuracy.sh [make_case.py 的参数...]
#
# 例：INSTALL=<custom opp 安装目录> CASE_DIR=<用例目录> bash run_accuracy.sh --dtype bf16 --gate gk --T 256
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
INSTALL=${INSTALL:?set INSTALL to the custom opp install root (contains vendors/)}
CASE_DIR=${CASE_DIR:?set CASE_DIR to a writable case directory}
VENDOR="$INSTALL/vendors/fla_npu_transformer"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

echo "== 1/4 生成用例"
python3 "$HERE/make_case.py" --dir "$CASE_DIR" "$@"

echo "== 2/4 编译 aclnn 取数程序"
g++ -std=c++17 -O2 "$HERE/test_aclnn_chunk_delta_h_bwd_preprocess.cpp" -o "$HERE/run_case" \
  -I"$VENDOR/op_api/include" \
  -I"$CANN/include" \
  -L"$VENDOR/op_api/lib" -lcust_opapi \
  -L"$CANN/lib64" -lascendcl -lnnopbase -lpthread -ldl

echo "== 3/4 运行"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:$CANN/lib64:${LD_LIBRARY_PATH:-}"
"$HERE/run_case" $(cat "$CASE_DIR/run_case_args.txt")

echo "== 4/4 与 CPU 标杆比对"
python3 "$HERE/compare.py" --dir "$CASE_DIR" "${EXTRA_COMPARE_ARGS:-}"
