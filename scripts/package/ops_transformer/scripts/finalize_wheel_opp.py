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
import importlib.util
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


def _load_abi_helper(package_dir: Path):
    helper_path = package_dir / "_op_abi.py"
    spec = importlib.util.spec_from_file_location("fla_npu_run_abi", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load wheel ABI helper: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _load_set_env_writer(package_dir: Path):
    helper_path = package_dir / "_opp_set_env.py"
    if not helper_path.is_file():
        raise RuntimeError(
            "The installed wheel is missing its OPP set_env writer: "
            f"{helper_path}. Install a matching complete wheel before applying "
            "this run package."
        )
    spec = importlib.util.spec_from_file_location("fla_npu_run_opp_set_env", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load wheel OPP set_env writer: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def validate_source_vendor(package_dir: Path, source_vendor: Path) -> None:
    """Reject partial/cross-version base replacement before any wheel write."""

    package_dir = package_dir.expanduser().resolve()
    helper = _load_abi_helper(package_dir)
    helper.validate_vendor_against_snapshot(
        source_vendor.expanduser().resolve(),
        package_dir / helper.BASE_CONTRACT_SNAPSHOT_NAME,
        require_exact=True,
    )


def _write_set_env(vendor_dir: Path, package_dir=None) -> None:
    if package_dir is None:
        package_dir = Path(vendor_dir).resolve().parents[2]
    _load_set_env_writer(Path(package_dir).resolve()).write_set_env(vendor_dir)


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

    helper = _load_abi_helper(package_dir)
    helper.validate_vendor_against_snapshot(
        vendor_dir,
        package_dir / helper.BASE_CONTRACT_SNAPSHOT_NAME,
        require_exact=True,
    )

    conflicting_alias = custom_opapi.with_name("libopapi.so")
    if conflicting_alias.exists() or conflicting_alias.is_symlink():
        conflicting_alias.unlink()

    _write_set_env(vendor_dir, package_dir)
    return _refresh_record(package_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--vendor-name", default=DEFAULT_VENDOR_DIR)
    parser.add_argument(
        "--check-source-vendor",
        type=Path,
        help="validate a complete run-package vendor before modifying the wheel",
    )
    args = parser.parse_args()

    if args.check_source_vendor is not None:
        validate_source_vendor(args.package_dir, args.check_source_vendor)
        print("Source vendor matches the installed wheel base ABI.")
        return 0

    record = finalize_wheel_opp(args.package_dir, args.vendor_name)
    print(f"Finalized wheel OPP and refreshed {record.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
