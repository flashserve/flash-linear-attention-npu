#!/usr/bin/env bash
set -Eeuo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1
source "$BASE/publish_env.sh"
load_publish_env obs-only
export no_proxy="${no_proxy:+$no_proxy,}mirrors.huaweicloud.com,obs.cn-north-4.myhuaweicloud.com"
export NO_PROXY="$no_proxy"
case "$(uname -m)" in aarch64) arch=arm ;; x86_64) arch=x86 ;; *) exit 2 ;; esac
PYTHON=${FLA_PKG_PYTHON:-$BASE/.venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=/usr/bin/python3
export PYTHONPATH="$BASE/.deps${PYTHONPATH:+:$PYTHONPATH}"
args=(--plan "${1:?plan required}" --arch "$arch"
      --timeout "${FLA_PKG_PUBLIC_VERIFY_TIMEOUT:-21600}"
      --interval "${FLA_PKG_LATEST_INTERVAL:-60}")
[[ -z "${2:-}" ]] || args+=(--done-file "$2")
exec "$PYTHON" "$BASE/publish_latest.py" "${args[@]}"
