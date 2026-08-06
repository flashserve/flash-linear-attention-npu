from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_OPP_ROOT = PACKAGE_DIR / "opp"


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


def _write_vendors_config(vendors_root: Path, vendor_dirs: list[Path]) -> None:
    vendor_names = sorted(path.name for path in vendor_dirs)
    if not vendor_names:
        raise RuntimeError(f"No vendor directories found under {vendors_root}")
    (vendors_root / "config.ini").write_text(
        f"load_priority={','.join(vendor_names)}\n",
        encoding="utf-8",
    )


def install_opp(install_path: Path, force: bool = False) -> None:
    vendors_src = PACKAGE_OPP_ROOT / "vendors"
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
        op_api_alias = vendor_dst / "op_api" / "lib" / "libopapi.so"
        if op_api_alias.exists() or op_api_alias.is_symlink():
            op_api_alias.unlink()
        _write_set_env(vendor_dst)
        copied.append(vendor_dst)

    if not copied:
        raise RuntimeError(f"No vendor directories found under {vendors_src}")
    _write_vendors_config(vendors_dst, copied)

    for vendor_dir in copied:
        print(f"Installed OPP vendor: {vendor_dir}")
    print(f"To use this external OPP location, set FLA_NPU_OPP_PATH={install_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy embedded FLA NPU OPP files to an external path.")
    parser.add_argument("--install-path", required=True, type=Path, help="Target OPP root. vendors/ will be created below it.")
    parser.add_argument("--force", action="store_true", help="Replace an existing target vendor directory.")
    args = parser.parse_args()

    install_opp(args.install_path, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
