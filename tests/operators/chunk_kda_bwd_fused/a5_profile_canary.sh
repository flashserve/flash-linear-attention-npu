#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kernel="${KDA_BWD_PROFILE_KERNEL:-c}"
heads="${KDA_BWD_PROFILE_HEADS:-8}"
seqlen="${KDA_BWD_PROFILE_SEQLEN:-1024}"
value_dim="${KDA_BWD_PROFILE_VALUE_DIM:-128}"
varlen_arg=()
if [[ "${KDA_BWD_PROFILE_VARLEN:-0}" == "1" ]]; then
    varlen_arg=(--varlen)
fi

python "${repo_root}/tests/operators/chunk_kda_bwd_fused/a2_canary.py" \
    --kernel "${kernel}" \
    --heads "${heads}" \
    --seqlen "${seqlen}" \
    --value-dim "${value_dim}" \
    --warmup 0 \
    --repeat 1 \
    "${varlen_arg[@]}"
