#!/usr/bin/env bash
# Source from an entrypoint after setting BASE to its installation directory.

load_publish_env() {
    local env_file=${FLA_PKG_ENV_FILE:-${BASE:?BASE is required}/.publish.env}
    if [[ -n "${FLA_PKG_ENV_FILE:-}" && ! -r "$env_file" ]]; then
        printf '[publish][ERROR] configuration is unreadable: %s\n' "$env_file" >&2
        return 2
    fi
    if [[ -f "$env_file" ]]; then
        set -a
        # This is a trusted, operator-owned shell configuration, not build input.
        source "$env_file" || return
        set +a
    fi
    local variable encoded decoded
    for variable in OBS_ACCESS_KEY OBS_SECRET_KEY GITHUB_TOKEN; do
        encoded="${variable}_B64"
        if [[ -z "${!variable:-}" && -n "${!encoded:-}" ]]; then
            decoded=$(printf '%s' "${!encoded}" | base64 -d) || {
                printf '[publish][ERROR] invalid base64 in %s\n' "$encoded" >&2
                return 2
            }
            printf -v "$variable" '%s' "$decoded"
        fi
        [[ -z "${!variable:-}" ]] || export "$variable"
        unset "$encoded"
    done
    unset FLA_PKG_GH_TOKEN
    if [[ "${1:-}" == obs-only ]]; then
        unset GITHUB_TOKEN
    fi
}
