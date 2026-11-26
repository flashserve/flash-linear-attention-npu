#!/usr/bin/env bash
# Run inside the standard CI container with its CANN environment and assigned card.
set -eo pipefail
platform="$1"
day="$2"
case "$platform" in a2|a5) ;; *) exit 2 ;; esac
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
index="https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/flash-linear-attention-npu-${platform}/"
stamp="${day//-/}"
python3 -m pip install --no-deps --no-index --no-cache-dir --pre --find-links "$index" \
    "flash-linear-attention-npu-${platform}==0.dev${stamp}"
vendor="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("fla_npu").origin).parent / "opp/vendors/fla_npu_transformer")')"
source "$vendor/bin/set_env.bash"
python3 ci/kda_model/run.py --platform "$platform" --date "$day" --report "kda-${platform}.json"
