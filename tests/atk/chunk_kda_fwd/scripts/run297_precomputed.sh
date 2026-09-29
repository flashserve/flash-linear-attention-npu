#!/bin/bash
# case 297 (idx 47) 精度：CPU 金标走预计算加载（/tmp/golden297，287 已逐位认证），
# 其余链路（NPU DUT、混合容差比较、报告）与正式 accuracy 完全一致。
# 说明：run_test_cpu.sh 硬编码 -p "./executor_${OP}.py"，无法换执行器，
# 因此本脚本直调 atk 并把 -p 指向 executor_chunk_kda_fwd_precomputed.py。
# 金标预先生成：gen_kda_golden_stream.py --case-id 297 ... --out-dir /tmp/golden297
# （见 ../README.md "case 297 预计算金标" 一节）。
set -u
# 本脚本位于 tests/atk/chunk_kda_fwd/scripts/，算子目录为其上一级
OP_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$OP_DIR"

# 环境默认值可用环境变量覆盖（交付机：CANN 9.1.0-beta.3 + /workspace/venv）
CANN_ENV_SCRIPT=${CANN_ENV_SCRIPT:-/usr/local/Ascend/cann/set_env.sh}
ATK_VENV_BIN=${ATK_VENV_BIN:-/workspace/venv/bin}
source "$CANN_ENV_SCRIPT" >/dev/null 2>&1
export PATH="$ATK_VENV_BIN:$PATH"
VD=$(python3 -c "import fla_npu, os; print(os.path.join(os.path.dirname(fla_npu.__file__), 'opp', 'vendors', 'fla_npu_transformer'))" 2>/dev/null)
if [ -z "$VD" ]; then
  VD=$(ls -d "$ATK_VENV_BIN"/../lib*/python3.11/site-packages/fla_npu/opp/vendors/fla_npu_transformer 2>/dev/null | head -1)
fi
export ASCEND_CUSTOM_OPP_PATH=$(dirname $(dirname $VD)):$VD
export KDA_GOLDEN_PRECOMPUTED_DIR=${KDA_GOLDEN_PRECOMPUTED_DIR:-/tmp/golden297}
export KDA_ATK_TRACE_SEED=1

ATK=$(command -v atk)
echo "[run297] start $(date '+%F %T')  ATK=$ATK  golden=$KDA_GOLDEN_PRECOMPUTED_DIR"
timeout --foreground -k 60 10800 "$ATK" node --name npu_dut --backend npu --devices 0 \
    --output_path ./atk_output/accuracy \
  node --name cpu_golden --backend cpu \
    --output_path ./atk_output/accuracy \
  task \
    -c ./atk_chunk_kda_fwd.json \
    --task accuracy \
    --bm_device cpu \
    -p ./executor_chunk_kda_fwd_precomputed.py \
    -s 47 -e 48 \
    --gm_init_flag \
    -to 14400 2>&1
rc=$?
echo "[run297] atk exit=$rc $(date '+%F %T')"
# 清理可能残留的 atk 进程（方括号防 pgrep 自匹配）
p=$(pgrep -f "atk nod[e]|run_test_cpu[.]sh|sqlite_we[b]|celery.*atk_outpu[t]")
[ -z "$p" ] || kill -KILL $p 2>/dev/null
exit $rc
