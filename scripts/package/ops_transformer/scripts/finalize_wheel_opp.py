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


DIST_INFO_GLOB = "flash_linear_attention_npu-*.dist-info"
DEFAULT_VENDOR_DIR = "fla_npu_transformer"


def _write_set_env(vendor_dir: Path) -> None:
    bin_dir = vendor_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    set_env = bin_dir / "set_env.bash"
    set_env.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                '_FLA_NPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                '_FLA_NPU_VENDOR_DIR="$(cd "${_FLA_NPU_SCRIPT_DIR}/.." && pwd)"',
                '_FLA_NPU_OPP_ROOT="$(cd "${_FLA_NPU_VENDOR_DIR}/../.." && pwd)"',
                "_fla_npu_prepend_path() {",
                '    local name="$1"',
                '    local value="$2"',
                '    local current=""',
                '    if [[ -v "${name}" ]]; then',
                '        current="${!name}"',
                "    fi",
                '    case ":${current}:" in',
                '        *":${value}:"*) return 0 ;;',
                "    esac",
                '    printf -v "${name}" "%s" "${value}${current:+:${current}}"',
                '    export "${name}"',
                "}",
                '_fla_npu_prepend_path ASCEND_CUSTOM_OPP_PATH "${_FLA_NPU_VENDOR_DIR}"',
                '_fla_npu_prepend_path ASCEND_CUSTOM_OPP_PATH "${_FLA_NPU_OPP_ROOT}"',
                '_fla_npu_prepend_path LD_LIBRARY_PATH "${_FLA_NPU_VENDOR_DIR}/op_api/lib"',
                'export FLA_NPU_OPP_PATH="${_FLA_NPU_OPP_ROOT}"',
                'export FLA_NPU_OP_API_LIB="${_FLA_NPU_VENDOR_DIR}/op_api/lib/libcust_opapi.so"',
                "unset -f _fla_npu_prepend_path",
                "unset _FLA_NPU_SCRIPT_DIR _FLA_NPU_VENDOR_DIR _FLA_NPU_OPP_ROOT",
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
    opp_root = package_dir / "opp"
    opp_prefix = f"{package_dir.name}/opp/"

    with record.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    rows = [row for row in rows if row and not row[0].startswith(opp_prefix)]
    for path in sorted(opp_root.rglob("*")):
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
        raise RuntimeError(f"Wheel OPP is missing custom op_api library: {custom_opapi}")

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
