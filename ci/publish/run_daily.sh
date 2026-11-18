#!/usr/bin/env bash
set -Eeuo pipefail

BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
umask 002
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1

provided_plan=
prepare_plan=
latest_head=0
while (( $# )); do
    case "$1" in
        --plan) provided_plan=${2:?missing plan}; shift 2 ;;
        --prepare-plan) prepare_plan=${2:?missing output}; shift 2 ;;
        --latest) latest_head=1; shift ;;
        *) printf 'Unknown daily option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$latest_head" == 1 && -z "$prepare_plan" ]]; then
    printf '%s\n' '--latest 必须与 --prepare-plan 配合，生成一份双机共用计划' >&2
    exit 2
fi

mkdir -p "$BASE/logs"
exec >>"$BASE/logs/cron-daily.log" 2>&1

# Credentials and optional host settings stay outside version control.
source "$BASE/publish_env.sh"
load_publish_env

# A deployment may provide a proxy script; plain process proxy variables work too.
PROXY_SCRIPT=${FLA_PKG_PROXY_SCRIPT:-}
if [[ -n "$PROXY_SCRIPT" && ! -r "$PROXY_SCRIPT" ]]; then
    printf '[daily][ERROR] proxy script is unreadable: %s\n' "$PROXY_SCRIPT" >&2
    exit 2
fi
if [[ -r "$PROXY_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$PROXY_SCRIPT"
    if declare -F _set_proxy_env >/dev/null; then
        _set_proxy_env
        proxy_status=000
        if command -v curl >/dev/null 2>&1; then
            proxy_status=$(curl -s --max-time 10 -o /dev/null \
                -w '%{http_code}' https://api.github.com/rate_limit || true)
        fi
        case "$proxy_status" in
            2??|3??)
                unset FLA_PKG_PROXY_DISABLED
                printf '[daily] GitHub proxy healthy and enabled via %s\n' "$PROXY_SCRIPT"
                ;;
            *)
                if declare -F _unset_proxy_env >/dev/null; then
                    _unset_proxy_env
                fi
                export FLA_PKG_PROXY_DISABLED=1
                printf '[daily][WARN] GitHub proxy unhealthy (HTTP %s); fallback to direct\n' \
                    "$proxy_status" >&2
                ;;
        esac
    fi
fi
export no_proxy="${no_proxy:+$no_proxy,}mirrors.huaweicloud.com,obs.cn-north-4.myhuaweicloud.com"
export NO_PROXY="$no_proxy"

case "$(uname -m)" in
    aarch64) daily_arch=arm ;;
    x86_64)  daily_arch=x86 ;;
    *)
        printf '[daily][ERROR] 不支持的宿主架构: %s\n' "$(uname -m)" >&2
        exit 2
        ;;
esac

printf '[daily] policy: daily -> OBS nightly only; GitHub Release disabled\n'
printf '[daily] host=%s arch=%s\n' "$(hostname)" "$daily_arch"

cd "$BASE"
PYTHON=${FLA_PKG_PYTHON:-$BASE/.venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=/usr/bin/python3
if [[ -d "$BASE/.deps" ]]; then
    export PYTHONPATH="$BASE/.deps${PYTHONPATH:+:$PYTHONPATH}"
fi

plan_date=$(date '+%Y%m%d')
plan_file=${provided_plan:-${prepare_plan:-$BASE/plans/$plan_date.json}}
mkdir -p "$BASE/plans"

if [[ -z "$prepare_plan" ]]; then
    exec 8>"$BASE/.run_daily.lock"
    flock -n 8 || { printf '[daily] another daily run is active\n'; exit 75; }
fi
if [[ -z "$provided_plan" ]]; then
    plan_args=(--date "$plan_date" --cutoff-hour 1 --output "$plan_file")
    read -r -a candidates <<< "${FLA_PKG_DAILY_BRANCHES:-main v26.10.0 v26.11.0}"
    for candidate in "${candidates[@]}"; do plan_args+=(--branch "$candidate"); done
    (( latest_head )) && plan_args+=(--latest)
    printf '[daily] resolve plan: %s latest_head=%s\n' "$plan_file" "$latest_head"
    "$PYTHON" "$BASE/daily_plan.py" "${plan_args[@]}"
fi
"$PYTHON" "$BASE/daily_plan.py" --read "$plan_file" --expect-date "$plan_date" >/dev/null
if [[ -n "$prepare_plan" ]]; then
    printf '[daily] plan prepared only: %s\n' "$plan_file"
    exit 0
fi
export FLA_PKG_DAILY_DATE="$plan_date"
"$PYTHON" "$BASE/publish_latest.py" --plan "$plan_file" --arch "$daily_arch" --check-plan

compile_args=(--build-type daily --arch "$daily_arch")
branch_count=0
while IFS=$'\t' read -r branch sha; do
    [[ -n "$branch" && -n "$sha" ]] || continue
    compile_args+=(--branch "$branch" --branch-sha "$branch=$sha")
    branch_count=$((branch_count + 1))
done < <("$PYTHON" "$BASE/daily_plan.py" --read "$plan_file" --expect-date "$plan_date")

# GitHub token is required only while creating the deterministic plan.  It is
# deliberately absent from Docker builds and OBS publishing.
unset GITHUB_TOKEN
unset FLA_PKG_GH_TOKEN

if (( branch_count == 0 )); then
    printf '[daily] all candidate release branches are formal; nothing to build\n'
    exit 0
fi

latest_done="$BASE/logs/$(basename "$plan_file" .json)-$daily_arch-$$.build.done"
"$BASE/watch_latest.sh" "$plan_file" "$latest_done" \
    >>"$BASE/logs/$(basename "$plan_file" .json)-$daily_arch-latest-watch.log" 2>&1 &
latest_pid=$!
trap 'if [[ -n "${latest_pid:-}" ]]; then kill "$latest_pid" 2>/dev/null || true; fi' EXIT
set +e
./run_compile.sh "${compile_args[@]}"
build_rc=$?
set -e
printf '%s\n' "$build_rc" >"$latest_done"
if wait "$latest_pid"; then latest_rc=0; else latest_rc=$?; fi
latest_pid=
if (( build_rc != 0 )); then
    printf '[daily][ERROR] build/publish failed; skip public matrix wait (exit=%s)\n' \
        "$build_rc" >&2
    exit "$build_rc"
fi
if (( latest_rc != 0 )); then
    printf '[daily][ERROR] latest 部分组合未成功；对应旧包已保留\n' >&2
    exit "$latest_rc"
fi

verify_enabled=${FLA_PKG_PUBLIC_VERIFY_ENABLED:-1}
verify_timeout=${FLA_PKG_PUBLIC_VERIFY_TIMEOUT:-21600}
verify_interval=${FLA_PKG_PUBLIC_VERIFY_INTERVAL:-300}
# The concurrent watcher has already finalized each successful slot separately.
if [[ "$verify_enabled" == 0 ]]; then
    printf '[daily] public mirror verification disabled by environment\n'
    exit 0
fi

verify_report="$BASE/logs/$(basename "$plan_file" .json)-$daily_arch-public-verify.json"
printf '[daily] wait for public matrix: branches=%s arches=arm,x86 platforms=all\n' \
    "$branch_count"
"$PYTHON" "$BASE/verify_daily_public.py" \
    --plan "$plan_file" \
    --timeout "$verify_timeout" \
    --interval "$verify_interval" \
    --report "$verify_report"
