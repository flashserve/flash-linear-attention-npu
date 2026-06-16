#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
retention_days="${CI_LOG_RETENTION_DAYS:-7}"

if [[ "${CI_LOG_CLEANUP_ENABLED:-true}" != "true" ]]; then
    echo "[CI] Log cleanup disabled."
    exit 0
fi

if ! [[ "$retention_days" =~ ^[0-9]+$ ]]; then
    echo "[CI][ERROR] CI_LOG_RETENTION_DAYS must be a non-negative integer: $retention_days" >&2
    exit 2
fi

declare -a cleanup_dirs=(
    "$repo_dir/log"
    "$repo_dir/logs"
    "$repo_dir/log_ut"
    "$repo_dir/output/log"
    "$repo_dir/output/logs"
    "$repo_dir/build/log"
    "$repo_dir/build/logs"
    "$repo_dir/build_out/log"
    "$repo_dir/build_out/logs"
    "${HOME:-/root}/ascend/log"
    "${HOME:-/root}/Ascend/log"
    "/var/log/ascend"
    "/var/log/npu"
)

if [[ -n "${CI_LOG_CLEANUP_DIRS:-}" ]]; then
    IFS=':' read -r -a extra_dirs <<< "$CI_LOG_CLEANUP_DIRS"
    cleanup_dirs+=("${extra_dirs[@]}")
fi

safe_dir() {
    local dir="$1"
    [[ -n "$dir" && "$dir" != "/" && "$dir" != "." && "$dir" != ".." ]]
}

echo "[CI] Cleaning CI logs older than ${retention_days} day(s)."
for dir in "${cleanup_dirs[@]}"; do
    if ! safe_dir "$dir" || [[ ! -d "$dir" ]]; then
        continue
    fi
    if deleted_files="$(find "$dir" -type f -mtime +"$retention_days" -print -delete 2>&1 | wc -l)"; then
        echo "[CI] Log cleanup dir: $dir, deleted files: $deleted_files"
    else
        echo "[CI][WARN] Failed to clean old log files in $dir" >&2
    fi
    if ! find "$dir" -mindepth 1 -type d -empty -mtime +"$retention_days" -delete 2>/dev/null; then
        echo "[CI][WARN] Failed to remove empty old log dirs in $dir" >&2
    fi
done
