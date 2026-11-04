#!/usr/bin/env bash
# ATK owns its assets, preflight and execution; CI only selects operators.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ -z "${CI_ATK_OPS:-}" || "$CI_ATK_OPS" == null ]]; then
    echo "[ATK CI] 无需验证的算子，跳过 ATK 精度阶段"
    exit 0
fi

IFS=',' read -r -a operators <<<"$CI_ATK_OPS"
if [[ ! "$CI_ATK_OPS" =~ ^[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*$ || ",${CI_ATK_OPS}," == *,null,* ]]; then
    echo "[ATK CI] 无效算子列表：${CI_ATK_OPS}；建议使用逗号分隔的 tests/atk 算子目录名" >&2
    exit 1
fi
for op in "${operators[@]}"; do
    echo "[ATK CI] 开始算子 ${op} 精度验证"
    bash tests/atk/run_test_cpu.sh \
        -op="$op" -scope=accuracy -ci=true \
        -npu_device_id="${NPU_DEVICE_ID:-0}" -soc="${SOC:-auto}"
done
