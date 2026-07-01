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
                'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                'VENDOR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"',
                'OPP_ROOT="$(cd "${VENDOR_DIR}/../.." && pwd)"',
                'export ASCEND_CUSTOM_OPP_PATH="${OPP_ROOT}:${VENDOR_DIR}:${ASCEND_CUSTOM_OPP_PATH}"',
                'export LD_LIBRARY_PATH="${VENDOR_DIR}/op_api/lib:${LD_LIBRARY_PATH}"',
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
