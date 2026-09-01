#!/bin/bash
set -e
cd "$(dirname "$0")"

# Same interpreter resolution as gen.sh: the caller may pass FLA_NPU_PYTHON or
# PYTHON, otherwise fall back to whatever 'python3' resolves to via PATH.
PY="${FLA_NPU_PYTHON:-${PYTHON:-python3}}"

FLA_NPU_PYTHON="$PY" bash gen.sh npu_custom.yaml
"$PY" setup.py bdist_wheel
"$PY" -m pip install ./dist/fla_npu-1.0.0-*.whl --force-reinstall --no-deps

# ASCEND_CUSTOM_OPP_PATH：CANN 在初始化时注册自定义 OPP；若 Python/ATK/Celery 等进程
# 会先初始化 CANN，需要在进程启动前设置该变量，否则可能出现 SelectBin 找不到 kernel
# （如 aclnnStatus=561103）。
_fla_npu_site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
_fla_npu_vendor="${_fla_npu_site}/fla_npu/opp/vendors/fla_npu_transformer"
echo ""
echo "[fla-npu] 若使用 fla_npu 的进程会先初始化 CANN（Python/ATK/Celery 等），请在启动前执行："
echo "[fla-npu]   export ASCEND_CUSTOM_OPP_PATH=\"${_fla_npu_vendor}:${_fla_npu_vendor}/op_api/lib:\${ASCEND_CUSTOM_OPP_PATH:-}\""
