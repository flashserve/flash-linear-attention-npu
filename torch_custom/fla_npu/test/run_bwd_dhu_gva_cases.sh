#!/bin/bash
# bwd_dhu GVA 双标杆：随机输入直接测，无需 example dump
#
# 用法:
#   TEST_DEVICE_ID=7 bash run_bwd_dhu_gva_cases.sh
#   BWD_HU_CASE=smoke_varlen_t256_v256 TEST_DEVICE_ID=7 bash run_bwd_dhu_gva_cases.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${TEST_DEVICE_ID:-5}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate wnc
source /data/zs/run/8.5/ascend-toolkit/set_env.sh
# 使用含 GVA kernel 的 vendor；安装后勿混用多层 ASCEND_CUSTOM_OPP_PATH
export ASCEND_CUSTOM_OPP_PATH=/data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer_transformer
export LD_LIBRARY_PATH=/data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer_transformer/op_api/lib:${LD_LIBRARY_PATH:-}

export TEST_DEVICE_ID="$DEVICE"
export BWD_HU_OUT_DIR="${BWD_HU_OUT_DIR:-$SCRIPT_DIR/bwd_dhu_out}"
export PYTHONUNBUFFERED=1

echo "device=$DEVICE out_dir=$BWD_HU_OUT_DIR"
cd "$SCRIPT_DIR"
python3 test_npu_bwd_dhu_gva.py
