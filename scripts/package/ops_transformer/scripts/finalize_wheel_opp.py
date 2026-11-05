#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Finalize an OPP overlay inside an installed fla_npu wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import stat
import tempfile
from pathlib import Path


# The wheel is published per product tier (flash-linear-attention-npu-a2/a3/a5)
# and a local build carries the same name, so match the family rather than the
# old base name: a run package installed over a tiered wheel used to find no
# dist-info at all and refuse to refresh RECORD.
DIST_INFO_GLOB = "flash_linear_attention_npu*.dist-info"
DEFAULT_VENDOR_DIR = "fla_npu_transformer"


def _write_set_env(vendor_dir: Path) -> None:
    bin_dir = vendor_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    set_env = bin_dir / "set_env.bash"
    set_env.write_text(
        "\n".join(
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
                '_FLA_NPU_LIB_ORDER=()',
                'for _FLA_NPU_ENTRY in "${_FLA_NPU_VENDOR_ORDER[@]}"; do',
                '    _FLA_NPU_LIB_ORDER+=("${_FLA_NPU_ENTRY}/op_api/lib/libcust_opapi.so")',
                "done",
                'export FLA_NPU_OP_API_LIBS="$(IFS=:; printf "%s" "${_FLA_NPU_LIB_ORDER[*]}")"',
                'if [[ -d "${_FLA_NPU_VENDORS_DIR}/fla_npu_transformer" ]]; then',
                '    export FLA_NPU_OP_API_LIB="${_FLA_NPU_VENDORS_DIR}/fla_npu_transformer/op_api/lib/libcust_opapi.so"',
                'elif [[ ${#_FLA_NPU_LIB_ORDER[@]} -gt 0 ]]; then',
                '    export FLA_NPU_OP_API_LIB="${_FLA_NPU_LIB_ORDER[0]}"',
                "else",
                '    unset FLA_NPU_OP_API_LIB',
                "fi",
                "unset -f _fla_npu_add_vendor_name _fla_npu_add_overlay _fla_npu_discover_overlays",
                "unset _FLA_NPU_SCRIPT_DIR _FLA_NPU_VENDOR_DIR _FLA_NPU_VENDORS_DIR _FLA_NPU_OPP_ROOT",
                "unset _FLA_NPU_VENDOR_ORDER _FLA_NPU_NAMES _FLA_NPU_NAME _FLA_NPU_LINE",
                "unset _FLA_NPU_CURRENT_PATHS _FLA_NPU_LIB_ORDER",
                "unset _FLA_NPU_ENTRY _FLA_NPU_EXTERNAL_PATHS _FLA_NPU_PATH_ORDER",
                "unset _FLA_NPU_OLD_PATH _FLA_NPU_NEW_PATH _FLA_NPU_HAD_PACKAGE_PATH",
                "",
            ]
        ),
        encoding="utf-8",
    )
    set_env.chmod(0o755)


def _record_digest(path: Path) -> tuple[str, str]:
    content = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}", str(len(content))


def _find_record(package_dir: Path) -> Path:
    dist_infos = sorted(package_dir.parent.glob(DIST_INFO_GLOB))
    if len(dist_infos) != 1:
        raise RuntimeError(
            f"Expected exactly one {DIST_INFO_GLOB} next to {package_dir}, "
            f"found {len(dist_infos)}"
        )
    record = dist_infos[0] / "RECORD"
    if not record.is_file():
        raise RuntimeError(f"Installed wheel RECORD is missing: {record}")
    return record


def _refresh_record(package_dir: Path) -> Path:
    record = _find_record(package_dir)
    site_root = package_dir.parent.resolve()
    package_roots = (package_dir / "opp",)
    package_prefixes = (f"{package_dir.name}/opp/",)

    with record.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    rows = [
        row
        for row in rows
        if row and not row[0].startswith(package_prefixes)
    ]
    for package_root in package_roots:
        for path in sorted(package_root.rglob("*")):
            if not (path.is_file() or path.is_symlink()):
                continue
            relative = path.relative_to(site_root).as_posix()
            digest, size = _record_digest(path)
            rows.append([relative, digest, size])

    record_mode = stat.S_IMODE(record.stat().st_mode)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=record.parent,
            prefix=".RECORD.",
            delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerows(rows)
        os.chmod(temp_name, record_mode)
        os.replace(temp_name, record)
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
    return record


def finalize_wheel_opp(package_dir: Path, vendor_name: str = DEFAULT_VENDOR_DIR) -> Path:
    package_dir = package_dir.expanduser().resolve()
    vendor_dir = package_dir / "opp" / "vendors" / vendor_name
    custom_opapi = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    if not custom_opapi.is_file():
        raise RuntimeError(f"Wheel is missing packaged custom op_api: {custom_opapi}")

    conflicting_alias = custom_opapi.with_name("libopapi.so")
    if conflicting_alias.exists() or conflicting_alias.is_symlink():
        conflicting_alias.unlink()

    _write_set_env(vendor_dir)
    return _refresh_record(package_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--vendor-name", default=DEFAULT_VENDOR_DIR)
    args = parser.parse_args()

    record = finalize_wheel_opp(args.package_dir, args.vendor_name)
    print(f"Finalized wheel OPP and refreshed {record.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
