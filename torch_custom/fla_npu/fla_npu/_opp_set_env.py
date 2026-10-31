"""Generate the OPP environment script without importing the fla_npu package."""

from __future__ import annotations

from pathlib import Path


def render_set_env_bash() -> str:
    """Return the single canonical ``set_env.bash`` implementation."""

    return "\n".join(
        [
            "#!/bin/bash",
            '_FLA_NPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"',
            '_FLA_NPU_VENDOR_DIR="$(cd "${_FLA_NPU_SCRIPT_DIR}/.." && pwd -P)"',
            '_FLA_NPU_VENDORS_DIR="$(cd "${_FLA_NPU_VENDOR_DIR}/.." && pwd -P)"',
            '_FLA_NPU_OPP_ROOT="$(cd "${_FLA_NPU_VENDORS_DIR}/.." && pwd -P)"',
            "_FLA_NPU_VENDOR_ORDER=()",
            "_fla_npu_add_vendor_name() {",
            '    local name="$1" candidate existing',
            '    case "${name}" in ""|"."|".."|*/*) return 0 ;; esac',
            '    candidate="$(cd "${_FLA_NPU_VENDORS_DIR}/${name}" 2>/dev/null && pwd -P)" || return 0',
            '    case "${candidate}" in "${_FLA_NPU_VENDORS_DIR}/"*) ;; *) return 0 ;; esac',
            '    for existing in "${_FLA_NPU_VENDOR_ORDER[@]}"; do',
            '        [[ "${existing}" == "${candidate}" ]] && return 0',
            "    done",
            '    _FLA_NPU_VENDOR_ORDER+=("${candidate}")',
            "}",
            "_fla_npu_add_overlay() {",
            '    local name="$1"',
            '    name="${name#"${name%%[![:space:]]*}"}"',
            '    name="${name%"${name##*[![:space:]]}"}"',
            '    [[ "${name}" =~ ^fla_npu_overlay_[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || return 0',
            '    _fla_npu_add_vendor_name "${name}"',
            "}",
            "_fla_npu_discover_overlays() {",
            "    local LC_ALL=C _FLA_NPU_ENTRY",
            '    for _FLA_NPU_ENTRY in "${_FLA_NPU_VENDORS_DIR}"/fla_npu_overlay_*; do',
            '        [[ -d "${_FLA_NPU_ENTRY}" ]] || continue',
            '        _fla_npu_add_overlay "${_FLA_NPU_ENTRY##*/}"',
            "    done",
            "}",
            'if [[ ${FLA_NPU_OVERLAY_VENDOR+x} ]]; then',
            '    IFS=\':\' read -r -a _FLA_NPU_NAMES <<< "${FLA_NPU_OVERLAY_VENDOR}"',
            '    for _FLA_NPU_NAME in "${_FLA_NPU_NAMES[@]}"; do',
            '        _fla_npu_add_overlay "${_FLA_NPU_NAME}"',
            "    done",
            "else",
            '    if [[ -f "${_FLA_NPU_VENDORS_DIR}/config.ini" ]]; then',
            '        while IFS= read -r _FLA_NPU_LINE || [[ -n "${_FLA_NPU_LINE}" ]]; do',
            '            _FLA_NPU_LINE="${_FLA_NPU_LINE#"${_FLA_NPU_LINE%%[![:space:]]*}"}"',
            '            case "${_FLA_NPU_LINE}" in',
            "                load_priority=*)",
            '                    IFS=\',\' read -r -a _FLA_NPU_NAMES <<< "${_FLA_NPU_LINE#load_priority=}"',
            '                    for _FLA_NPU_NAME in "${_FLA_NPU_NAMES[@]}"; do',
            '                        _fla_npu_add_overlay "${_FLA_NPU_NAME}"',
            "                    done",
            "                    break",
            "                    ;;",
            "            esac",
            '        done < "${_FLA_NPU_VENDORS_DIR}/config.ini"',
            "    fi",
            "    _fla_npu_discover_overlays",
            "fi",
            '_fla_npu_add_vendor_name "fla_npu_transformer"',
            # Preserve standalone external vendor exports without a FLA base.
            'if [[ ! -d "${_FLA_NPU_VENDORS_DIR}/fla_npu_transformer" ]]; then',
            '    case "${_FLA_NPU_VENDOR_DIR##*/}" in',
            '        fla_npu_overlay_*) ;;',
            '        *) _fla_npu_add_vendor_name "${_FLA_NPU_VENDOR_DIR##*/}" ;;',
            "    esac",
            "fi",
            '_FLA_NPU_OLD_PATH="${ASCEND_CUSTOM_OPP_PATH-}"',
            "_FLA_NPU_EXTERNAL_PATHS=()",
            "_FLA_NPU_HAD_PACKAGE_PATH=0",
            'IFS=\':\' read -r -a _FLA_NPU_CURRENT_PATHS <<< "${_FLA_NPU_OLD_PATH}"',
            'for _FLA_NPU_ENTRY in "${_FLA_NPU_CURRENT_PATHS[@]}"; do',
            '    [[ -z "${_FLA_NPU_ENTRY}" ]] && continue',
            '    case "${_FLA_NPU_ENTRY}" in',
            '        "${_FLA_NPU_OPP_ROOT}"|"${_FLA_NPU_OPP_ROOT}/"*)',
            "            _FLA_NPU_HAD_PACKAGE_PATH=1",
            "            continue",
            "            ;;",
            "    esac",
            '    _FLA_NPU_EXTERNAL_PATHS+=("${_FLA_NPU_ENTRY}")',
            "done",
            '_FLA_NPU_PATH_ORDER=("${_FLA_NPU_VENDOR_ORDER[@]}" "${_FLA_NPU_EXTERNAL_PATHS[@]}")',
            '_FLA_NPU_NEW_PATH="$(IFS=:; printf "%s" "${_FLA_NPU_PATH_ORDER[*]}")"',
            'if [[ ${_FLA_NPU_HAD_PACKAGE_PATH} -eq 1 && "${_FLA_NPU_OLD_PATH}" != "${_FLA_NPU_NEW_PATH}" ]]; then',
            '    printf "%s\\n" "[fla-npu] warning: normalized package-local ASCEND_CUSTOM_OPP_PATH to overlay-first vendor order." >&2',
            "fi",
            'export ASCEND_CUSTOM_OPP_PATH="${_FLA_NPU_NEW_PATH}"',
            'export FLA_NPU_OPP_PATH="${_FLA_NPU_OPP_ROOT}"',
            "_FLA_NPU_LIB_ORDER=()",
            'for _FLA_NPU_ENTRY in "${_FLA_NPU_VENDOR_ORDER[@]}"; do',
            '    _FLA_NPU_LIB_ORDER+=("${_FLA_NPU_ENTRY}/op_api/lib/libcust_opapi.so")',
            "done",
            'export FLA_NPU_OP_API_LIBS="$(IFS=:; printf "%s" "${_FLA_NPU_LIB_ORDER[*]}")"',
            'if [[ -d "${_FLA_NPU_VENDORS_DIR}/fla_npu_transformer" ]]; then',
            '    export FLA_NPU_OP_API_LIB="${_FLA_NPU_VENDORS_DIR}/fla_npu_transformer/op_api/lib/libcust_opapi.so"',
            'elif [[ ${#_FLA_NPU_LIB_ORDER[@]} -gt 0 ]]; then',
            '    export FLA_NPU_OP_API_LIB="${_FLA_NPU_LIB_ORDER[0]}"',
            "else",
            "    unset FLA_NPU_OP_API_LIB",
            "fi",
            "unset -f _fla_npu_add_vendor_name _fla_npu_add_overlay _fla_npu_discover_overlays",
            "unset _FLA_NPU_SCRIPT_DIR _FLA_NPU_VENDOR_DIR _FLA_NPU_VENDORS_DIR _FLA_NPU_OPP_ROOT",
            "unset _FLA_NPU_VENDOR_ORDER _FLA_NPU_NAMES _FLA_NPU_NAME _FLA_NPU_LINE",
            "unset _FLA_NPU_CURRENT_PATHS _FLA_NPU_LIB_ORDER",
            "unset _FLA_NPU_ENTRY _FLA_NPU_EXTERNAL_PATHS _FLA_NPU_PATH_ORDER",
            "unset _FLA_NPU_OLD_PATH _FLA_NPU_NEW_PATH _FLA_NPU_HAD_PACKAGE_PATH",
            "",
        ]
    )


def write_set_env(vendor_dir: Path) -> Path:
    """Write the canonical OPP environment script and make it executable."""

    set_env = Path(vendor_dir) / "bin" / "set_env.bash"
    set_env.parent.mkdir(parents=True, exist_ok=True)
    set_env.write_text(render_set_env_bash(), encoding="utf-8")
    set_env.chmod(0o755)
    return set_env
