from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_OPP_ROOT = PACKAGE_DIR / "opp"
DEFAULT_OVERLAY_NAME = "fla_npu_overlay_transformer"
OVERLAY_VENDOR_ENV = "FLA_NPU_OVERLAY_VENDOR"
BASE_VENDOR_NAME = "fla_npu_transformer"


def overlay_vendor_names() -> list[str]:
    """Overlay vendor names, highest priority first (env override supported)."""

    raw = os.environ.get(OVERLAY_VENDOR_ENV, DEFAULT_OVERLAY_NAME)
    return [name.strip() for name in raw.split(os.pathsep) if name.strip()]


def _vendors_root() -> Path:
    return PACKAGE_OPP_ROOT / "vendors"


def _write_set_env(vendor_dir: Path) -> None:
    bin_dir = vendor_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "set_env.bash").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                '_FLA_NPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                '_FLA_NPU_VENDOR_DIR="$(cd "${_FLA_NPU_SCRIPT_DIR}/.." && pwd)"',
                '_FLA_NPU_OPP_ROOT="$(cd "${_FLA_NPU_VENDOR_DIR}/../.." && pwd)"',
                "_fla_npu_prepend_path() {",
                '    local name="$1"',
                '    local value="$2"',
                "    local current=\"\"",
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
                'export FLA_NPU_OPP_PATH="${_FLA_NPU_OPP_ROOT}"',
                'export FLA_NPU_OP_API_LIB="${_FLA_NPU_VENDOR_DIR}/op_api/lib/libcust_opapi.so"',
                "unset -f _fla_npu_prepend_path",
                "unset _FLA_NPU_SCRIPT_DIR _FLA_NPU_VENDOR_DIR _FLA_NPU_OPP_ROOT",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _ordered_vendor_names(vendors_root: Path) -> list[str]:
    """Priority order: overlay vendors first, then the base vendor."""

    ordered: list[str] = []
    for name in [*overlay_vendor_names(), BASE_VENDOR_NAME]:
        if name in ordered:
            continue
        if (vendors_root / name).is_dir():
            ordered.append(name)
    # Keep any other vendor directories (do not silently drop them).
    for child in sorted(vendors_root.iterdir()):
        if child.is_dir() and child.name not in ordered:
            ordered.append(child.name)
    return ordered


def _write_vendors_config(vendors_root: Path) -> list[str]:
    vendor_names = _ordered_vendor_names(vendors_root)
    if not vendor_names:
        raise RuntimeError(f"No vendor directories found under {vendors_root}")
    (vendors_root / "config.ini").write_text(
        f"load_priority={','.join(vendor_names)}\n",
        encoding="utf-8",
    )
    return vendor_names


def _vendor_ops(vendor_dir: Path) -> list[str]:
    """Operator names listed in the vendor's ops-info (best effort)."""

    ops: list[str] = []
    for info in vendor_dir.glob("op_impl/ai_core/tbe/config/*/aic-*-ops-info.json"):
        try:
            ops.extend(json.loads(info.read_text(encoding="utf-8")).keys())
        except Exception:
            continue
    return sorted(set(ops))


def install_opp(install_path: Path, force: bool = False) -> None:
    vendors_src = _vendors_root()
    if not vendors_src.exists():
        raise FileNotFoundError(f"Embedded OPP vendors directory not found: {vendors_src}")

    install_path = install_path.expanduser().resolve()
    vendors_dst = install_path / "vendors"
    vendors_dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for vendor_src in vendors_src.iterdir():
        if not vendor_src.is_dir():
            continue
        vendor_dst = vendors_dst / vendor_src.name
        if vendor_dst.exists():
            if not force:
                raise FileExistsError(f"{vendor_dst} already exists. Use --force to replace it.")
            shutil.rmtree(vendor_dst)
        shutil.copytree(vendor_src, vendor_dst)
        custom_opapi = vendor_dst / "op_api" / "lib" / "libcust_opapi.so"
        if not custom_opapi.is_file():
            raise FileNotFoundError(f"Custom op_api library not found: {custom_opapi}")
        op_api_alias = vendor_dst / "op_api" / "lib" / "libopapi.so"
        if op_api_alias.exists() or op_api_alias.is_symlink():
            op_api_alias.unlink()
        _write_set_env(vendor_dst)
        copied.append(vendor_dst)

    if not copied:
        raise RuntimeError(f"No vendor directories found under {vendors_src}")
    _write_vendors_config(vendors_dst)

    for vendor_dir in copied:
        print(f"Installed OPP vendor: {vendor_dir}")
    print(
        "The external OPP is available to CANN/ACLNN after sourcing its "
        "set_env.bash. The fla_npu Python runtime always loads libcust_opapi.so "
        "from the vendor tree embedded in its installed package."
    )


def _resolve_overlay_source(source: Path) -> Path:
    """Accept a vendor dir, or a parent containing exactly one vendor dir."""

    source = source.expanduser().resolve()
    if (source / "op_api" / "lib" / "libcust_opapi.so").is_file():
        return source
    candidates = [
        child
        for child in sorted(source.iterdir())
        if child.is_dir() and (child / "op_api" / "lib" / "libcust_opapi.so").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No vendor directory (with op_api/lib/libcust_opapi.so) found under {source}"
        )
    raise RuntimeError(
        f"Ambiguous overlay source {source}: found {[c.name for c in candidates]}. "
        "Point --source at a single vendor directory."
    )


def install_overlay(source: Path, overlay_name: str | None = None, force: bool = False) -> Path:
    """Install an overlay vendor next to the base vendor, inside the package.

    The base vendor is left untouched. No ``RECORD`` rewrite happens: pip keeps
    its own manifest of the wheel, so the overlay is removed with
    ``--remove-overlay`` instead of ``pip uninstall``.
    """

    vendors_root = _vendors_root()
    vendors_root.mkdir(parents=True, exist_ok=True)
    overlay_name = overlay_name or overlay_vendor_names()[0]
    if overlay_name == BASE_VENDOR_NAME:
        raise RuntimeError(
            f"Refusing to install an overlay over the base vendor {BASE_VENDOR_NAME}: "
            "pass --overlay-name to keep both trees side by side."
        )
    if overlay_name in overlay_vendor_names():
        # Force this overlay to be searched first, ahead of other overlays.
        os.environ[OVERLAY_VENDOR_ENV] = os.pathsep.join(
            [overlay_name, *[n for n in overlay_vendor_names() if n != overlay_name]]
        )

    vendor_src = _resolve_overlay_source(source)
    vendor_dst = vendors_root / overlay_name
    if vendor_dst.exists():
        if not force:
            raise FileExistsError(
                f"{vendor_dst} already exists. Use --force to replace the overlay."
            )
        shutil.rmtree(vendor_dst)
    shutil.copytree(vendor_src, vendor_dst)

    custom_opapi = vendor_dst / "op_api" / "lib" / "libcust_opapi.so"
    if not custom_opapi.is_file():
        shutil.rmtree(vendor_dst, ignore_errors=True)
        raise FileNotFoundError(f"Custom op_api library not found in the overlay: {custom_opapi}")
    op_api_alias = vendor_dst / "op_api" / "lib" / "libopapi.so"
    if op_api_alias.exists() or op_api_alias.is_symlink():
        op_api_alias.unlink()
    _write_set_env(vendor_dst)

    order = _write_vendors_config(vendors_root)

    files = sum(1 for path in vendor_dst.rglob("*") if path.is_file())
    print(f"Installed overlay vendor: {vendor_dst}")
    print(f"  files copied : {files}")
    print(f"  operators    : {_vendor_ops(vendor_dst)}")
    print(f"  load_priority: {','.join(order)}")
    print(
        "  note: pip's RECORD is intentionally not rewritten (the overlay is not a "
        "pip-managed file set). Remove it with:\n"
        f"        python -m fla_npu.install_opp --remove-overlay --overlay-name {overlay_name}"
    )
    return vendor_dst


def remove_overlay(overlay_name: str | None = None) -> bool:
    """Remove an overlay vendor and refresh ``config.ini``. Base vendor untouched."""

    vendors_root = _vendors_root()
    overlay_name = overlay_name or overlay_vendor_names()[0]
    if overlay_name == BASE_VENDOR_NAME:
        raise RuntimeError(f"Refusing to remove the base vendor {BASE_VENDOR_NAME}.")
    vendor_dir = vendors_root / overlay_name
    if not vendor_dir.is_dir():
        print(f"No overlay vendor to remove: {vendor_dir}")
        return False
    shutil.rmtree(vendor_dir)
    order = _write_vendors_config(vendors_root)
    print(f"Removed overlay vendor: {vendor_dir}")
    print(f"  load_priority: {','.join(order)}")
    return True


def print_status() -> None:
    vendors_root = _vendors_root()
    config = vendors_root / "config.ini"
    print(f"package OPP root : {PACKAGE_OPP_ROOT}")
    print(f"vendors root     : {vendors_root}")
    print(f"config.ini       : {config.read_text(encoding='utf-8').strip() if config.is_file() else '<missing>'}")
    if not vendors_root.is_dir():
        print("  <no vendors installed>")
        return
    for child in sorted(vendors_root.iterdir()):
        if not child.is_dir():
            continue
        kind = "base" if child.name == BASE_VENDOR_NAME else "overlay"
        files = sum(1 for path in child.rglob("*") if path.is_file())
        print(f"  [{kind:7s}] {child.name}: {files} files, ops={_vendor_ops(child)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the FLA NPU OPP embedded in the installed package.")
    parser.add_argument("--install-path", type=Path, help="Target OPP root. vendors/ will be created below it.")
    parser.add_argument("--force", action="store_true", help="Replace an existing target vendor directory.")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Install --source as an overlay vendor inside the package (base vendor untouched).",
    )
    parser.add_argument(
        "--remove-overlay",
        action="store_true",
        help="Remove the overlay vendor from the package and refresh config.ini.",
    )
    parser.add_argument("--status", action="store_true", help="Show the embedded OPP overlay state.")
    parser.add_argument("--source", type=Path, help="Delta source: a vendor dir or a parent holding one.")
    parser.add_argument("--overlay-name", help=f"Overlay vendor name (default: {DEFAULT_OVERLAY_NAME}).")
    args = parser.parse_args()

    modes = [bool(args.install_path), args.overlay, args.remove_overlay, args.status]
    if sum(modes) != 1:
        parser.error("choose exactly one of --install-path / --overlay / --remove-overlay / --status")

    if args.install_path:
        install_opp(args.install_path, force=args.force)
    elif args.overlay:
        if args.source is None:
            parser.error("--overlay requires --source")
        install_overlay(args.source, overlay_name=args.overlay_name, force=args.force)
    elif args.remove_overlay:
        remove_overlay(overlay_name=args.overlay_name)
    else:
        print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
