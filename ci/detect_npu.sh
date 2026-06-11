#!/usr/bin/env bash
set -euo pipefail

mode="${1:---summary}"
target_id="${2:-}"

if ! command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi is not available" >&2
    exit 1
fi

raw="$(npu-smi info 2>/dev/null || true)"
if [[ -z "$raw" ]]; then
    echo "npu-smi returned no data" >&2
    exit 1
fi

declare -a ids=()
declare -A names=()
declare -A health=()
declare -A free=()
declare -A sink_locked=()

while IFS= read -r line; do
    if [[ "$line" =~ ^\|[[:space:]]*([0-9]+)[[:space:]]+([^[:space:]\|]*[A-Za-z][^[:space:]\|]*)[[:space:]]*\|[[:space:]]*([A-Za-z]+) ]]; then
        id="${BASH_REMATCH[1]}"
        ids+=("$id")
        names["$id"]="${BASH_REMATCH[2]}"
        health["$id"]="${BASH_REMATCH[3]}"
    elif [[ "$line" =~ No[[:space:]]running[[:space:]]processes[[:space:]]found[[:space:]]in[[:space:]]NPU[[:space:]]+([0-9]+) ]]; then
        free["${BASH_REMATCH[1]}"]=1
    fi
done <<< "$raw"

if [[ "${#ids[@]}" -eq 0 ]]; then
    echo "No NPU device was found in npu-smi output" >&2
    exit 1
fi

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

driver_sink_lock_path() {
    local id="$1"
    local lock_dir="${CI_NPU_DRIVER_LOCK_DIR:-/usr/local/Ascend/driver/lib64/driver}"
    echo "${lock_dir}/sink_file_mutex_${id}.cfg"
}

has_driver_sink_lock() {
    local id="$1"
    if is_truthy "${CI_IGNORE_NPU_DRIVER_SINK_LOCKS:-false}"; then
        return 1
    fi
    if ! command -v lslocks >/dev/null 2>&1; then
        return 1
    fi

    local lock_path
    lock_path="$(driver_sink_lock_path "$id")"
    [[ -e "$lock_path" ]] || return 1

    lslocks 2>/dev/null | awk -v path="$lock_path" '$NF == path { found = 1 } END { exit found ? 0 : 1 }'
}

for id in "${ids[@]}"; do
    if has_driver_sink_lock "$id"; then
        sink_locked["$id"]=1
    else
        sink_locked["$id"]=0
    fi
done

soc_from_name() {
    local name="$1"
    case "$name" in
        *910B*|*910b*) echo "ascend910b" ;;
        *910_93*|*910*93*) echo "ascend910_93" ;;
        *950*) echo "ascend950" ;;
        *) echo "unknown" ;;
    esac
}

select_device() {
    local candidates=()
    mapfile -t candidates < <(emit_candidates)
    if (( ${#candidates[@]} > 0 )); then
        echo "${candidates[0]}"
        return
    fi

    local id
    for id in "${ids[@]}"; do
        if [[ "${health[$id]}" == "OK" && "${free[$id]:-0}" == "1" ]]; then
            echo "$id"
            return
        fi
    done
    for id in "${ids[@]}"; do
        if [[ "${free[$id]:-0}" == "1" ]]; then
            echo "$id"
            return
        fi
    done
    for id in "${ids[@]}"; do
        if [[ "${health[$id]}" == "OK" ]]; then
            echo "$id"
            return
        fi
    done
    echo "${ids[0]}"
}

is_candidate_allowed() {
    local id="$1"
    if [[ "${sink_locked[$id]:-0}" == "1" ]]; then
        return 1
    fi
    if [[ "${free[$id]:-0}" == "1" ]]; then
        return 0
    fi
    is_truthy "${CI_ALLOW_BUSY_NPU:-false}"
}

emit_candidates() {
    local id
    for id in "${ids[@]}"; do
        if [[ "${health[$id]}" == "OK" && "${free[$id]:-0}" == "1" ]] && is_candidate_allowed "$id"; then
            echo "$id"
        fi
    done
    if ! is_truthy "${CI_REQUIRE_HEALTHY_NPU:-false}"; then
        for id in "${ids[@]}"; do
            if [[ "${health[$id]}" != "OK" && "${free[$id]:-0}" == "1" ]] && is_candidate_allowed "$id"; then
                echo "$id"
            fi
        done
    fi
    if is_truthy "${CI_ALLOW_BUSY_NPU:-false}"; then
        for id in "${ids[@]}"; do
            if [[ "${health[$id]}" == "OK" && "${free[$id]:-0}" != "1" ]] && is_candidate_allowed "$id"; then
                echo "$id"
            fi
        done
    fi
    if is_truthy "${CI_ALLOW_BUSY_NPU:-false}" && ! is_truthy "${CI_REQUIRE_HEALTHY_NPU:-false}"; then
        for id in "${ids[@]}"; do
            if [[ "${health[$id]}" != "OK" && "${free[$id]:-0}" != "1" ]] && is_candidate_allowed "$id"; then
                echo "$id"
            fi
        done
    fi
}

emit_env_for() {
    local id="$1"
    if [[ -z "$id" || -z "${names[$id]+x}" ]]; then
        echo "Unknown NPU device: ${id:-<empty>}" >&2
        exit 2
    fi
    echo "NPU_AVAILABLE=1"
    echo "NPU_SELECTED_DEVICE=$id"
    echo "NPU_SELECTED_NAME=${names[$id]}"
    echo "NPU_SELECTED_HEALTH=${health[$id]}"
    echo "NPU_SELECTED_FREE=${free[$id]:-0}"
    echo "NPU_SOC=$(soc_from_name "${names[$id]}")"
}

selected="$(select_device)"
selected_soc="$(soc_from_name "${names[$selected]}")"
selected_free="${free[$selected]:-0}"

case "$mode" in
    --first-free|--selected)
        echo "$selected"
        ;;
    --soc)
        echo "$selected_soc"
        ;;
    --env)
        emit_env_for "$selected"
        ;;
    --env-for)
        emit_env_for "$target_id"
        ;;
    --candidates)
        emit_candidates
        ;;
    --summary)
        echo "Detected NPU devices:"
        for id in "${ids[@]}"; do
            echo "  - id=$id name=${names[$id]} health=${health[$id]} free=${free[$id]:-0} sink_locked=${sink_locked[$id]:-0} soc=$(soc_from_name "${names[$id]}")"
        done
        echo "Selected NPU: id=$selected name=${names[$selected]} health=${health[$selected]} free=$selected_free soc=$selected_soc"
        ;;
    *)
        echo "Usage: $0 [--summary|--env|--env-for <id>|--candidates|--first-free|--selected|--soc]" >&2
        exit 2
        ;;
esac
