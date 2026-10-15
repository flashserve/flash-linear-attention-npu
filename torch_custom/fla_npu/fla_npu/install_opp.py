from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import warnings
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_OPP_ROOT = PACKAGE_DIR / "opp"
DEFAULT_OVERLAY_NAME = "fla_npu_overlay_transformer"
# Any vendor directory with this prefix is an overlay: it outranks the base
# vendor in ``config.ini`` and is auto-discovered by the loader.
OVERLAY_VENDOR_PREFIX = "fla_npu_overlay_"
OVERLAY_VENDOR_ENV = "FLA_NPU_OVERLAY_VENDOR"
BASE_VENDOR_NAME = "fla_npu_transformer"
_VENDOR_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_OVERLAY_VENDOR_NAME_RE = re.compile(
    rf"{re.escape(OVERLAY_VENDOR_PREFIX)}[A-Za-z0-9][A-Za-z0-9_.-]*\Z"
)


def _is_overlay_name(name: str) -> bool:
    return bool(_OVERLAY_VENDOR_NAME_RE.fullmatch(name))


def overlay_vendor_names() -> list[str]:
    """Overlay vendor names, highest priority first (env override supported).

    ``FLA_NPU_OVERLAY_VENDOR`` wins when set; otherwise the overlays recorded in
    ``config.ini`` (plus any ``fla_npu_overlay_*`` directory) are used, matching
    what the loader auto-discovers at import time.
    """

    raw = os.environ.get(OVERLAY_VENDOR_ENV)
    if raw is not None:
        names: list[str] = []
        for candidate in raw.split(os.pathsep):
            name = candidate.strip()
            if not name:
                continue
            if not _is_overlay_name(name):
                warnings.warn(
                    f"Ignoring invalid {OVERLAY_VENDOR_ENV} entry {name!r}; overlay "
                    f"names must match {OVERLAY_VENDOR_PREFIX}<name>.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if name not in names:
                names.append(name)
        return names
    vendors_root = _vendors_root()
    names: list[str] = []

    def _add(candidate: str) -> None:
        name = candidate.strip()
        if not name or name == BASE_VENDOR_NAME or name in names:
            return
        if _is_overlay_name(name) and (vendors_root / name).is_dir():
            names.append(name)

    try:
        for line in (vendors_root / "config.ini").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("load_priority="):
                for entry in line.split("=", 1)[1].split(","):
                    _add(entry)
                break
    except OSError:
        pass
    try:
        for child in sorted(vendors_root.iterdir()):
            if child.is_dir():
                _add(child.name)
    except OSError:
        pass
    return names


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
    """Priority order: overlay vendors first, then the base vendor, then the rest.

    Overlay *directories* must outrank the base vendor even when they were
    installed under a name that is not (yet) in ``FLA_NPU_OVERLAY_VENDOR``;
    otherwise ``config.ini`` would pin the base vendor first and the freshly
    installed overlay would silently lose priority.
    """

    ordered: list[str] = []
    for name in [*overlay_vendor_names(), BASE_VENDOR_NAME]:
        if name in ordered:
            continue
        if (vendors_root / name).is_dir():
            ordered.append(name)
    # Keep any other vendor directories (do not silently drop them), but put
    # overlay-looking ones ahead of the base vendor.
    for child in sorted(vendors_root.iterdir()):
        if not child.is_dir() or child.name in ordered:
            continue
        if not _VENDOR_NAME_RE.fullmatch(child.name):
            warnings.warn(
                f"Ignoring vendor directory with unsafe name {child.name!r}; it cannot "
                "be represented safely in config.ini load_priority.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if _is_overlay_name(child.name) and BASE_VENDOR_NAME in ordered:
            ordered.insert(ordered.index(BASE_VENDOR_NAME), child.name)
        else:
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


def _validate_overlay_name(overlay_name: str) -> str:
    """Reject names that could escape ``opp/vendors`` or hit a non-overlay tree.

    ``Path("a") / "/abs"`` yields ``/abs``, and ``Path("a") / "../b"`` escapes the
    parent, so an unvalidated name would let ``rmtree`` delete an arbitrary
    absolute path.  A trailing separator or ``./`` would also slip past a plain
    ``== BASE_VENDOR_NAME`` comparison while still resolving to the base vendor.
    Only a single, plain component carrying the overlay prefix is accepted.
    """

    name = (overlay_name or "").strip()
    if not _is_overlay_name(name):
        raise RuntimeError(
            f"Refusing overlay name {overlay_name!r}: it must match "
            f"{OVERLAY_VENDOR_PREFIX}<name> using only ASCII letters, digits, '.', '_' "
            "or '-'. Directory separators, absolute paths and config delimiters are not allowed."
        )
    return name


def _default_overlay_name() -> str:
    """Return the documented default, independent of installed overlay order."""

    return DEFAULT_OVERLAY_NAME


def install_overlay(source: Path, overlay_name: str | None = None, force: bool = False) -> Path:
    """Install an overlay vendor next to the base vendor, inside the package.

    The base vendor is left untouched. No ``RECORD`` rewrite happens: pip keeps
    its own manifest of the wheel, so the overlay is removed with
    ``--remove-overlay`` instead of ``pip uninstall``.
    """

    vendors_root = _vendors_root()
    vendors_root.mkdir(parents=True, exist_ok=True)
    overlay_name = _validate_overlay_name(overlay_name or _default_overlay_name())
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
    overlay_name = _validate_overlay_name(overlay_name or _default_overlay_name())
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
    active = set(overlay_vendor_names())
    env_set = OVERLAY_VENDOR_ENV in os.environ
    for child in sorted(vendors_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == BASE_VENDOR_NAME:
            kind = "base"
        elif child.name in active:
            kind = "overlay"
        else:
            kind = "overlay?"  # installed but not part of the active load order
        files = sum(1 for path in child.rglob("*") if path.is_file())
        print(f"  [{kind:8s}] {child.name}: {files} files, ops={_vendor_ops(child)}")
    if not env_set:
        installed = sorted(
            child.name
            for child in vendors_root.iterdir()
            if child.is_dir() and _is_overlay_name(child.name)
        )
        if len(installed) > 1:
            print(
                "  hint: multiple overlays installed; the load order is taken from "
                f"config.ini. Cite them explicitly with "
                f'{OVERLAY_VENDOR_ENV}="{os.pathsep.join(installed)}" to pin the priority.'
            )


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
