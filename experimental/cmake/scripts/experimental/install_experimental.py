#!/usr/bin/env python3
"""Install an experimental run package without touching a stable OPP or wheel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile


VENDOR = "fla_npu_experimental_transformer"
MANIFEST = "experimental-build-manifest.json"


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _protected_roots() -> list[Path]:
    roots = [Path("/usr/local/Ascend")]
    for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH",
                 "FLA_NPU_OPP_PATH", "FLA_NPU_EXPERIMENTAL_OPP_PATH"):
        roots.extend(Path(item) for item in os.environ.get(name, "").split(os.pathsep) if item)
    # Locate the stable wheel without importing its runtime or loading a library.
    spec = importlib.util.find_spec("fla_npu")
    if spec is not None and spec.origin:
        roots.append(Path(spec.origin).resolve().parent)
    return [root.resolve() for root in roots]


def _validate_destination(raw_path: str) -> Path:
    path = Path(raw_path)
    if not raw_path or not path.is_absolute():
        raise ValueError("--install-path must name an absolute empty directory")
    if ".." in path.parts:
        raise ValueError("--install-path must not contain '..'")
    # Resolve only after examining every lexical ancestor; resolving first would
    # conceal an alias into a stable wheel or the toolkit installation.
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"install path must not contain a symlink: {ancestor}")
    destination = path.resolve()
    if destination == destination.parent:
        raise ValueError("cannot install into a filesystem root")
    for root in _protected_roots():
        if _within(destination, root):
            raise ValueError(f"install path is inside an active/stable OPP or toolkit: {root}")
    for ancestor in (destination, *destination.parents):
        if ((ancestor / "vendors").exists()
                or (ancestor.name == "fla_npu" and (ancestor / "__init__.py").exists())):
            raise ValueError(f"install path is inside an existing OPP or stable package: {ancestor}")
    if not destination.parent.is_dir():
        raise ValueError("the install path's parent directory must already exist")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("install destination must be absent or an empty directory")
    return destination


def _validate_payload(package_root: Path) -> Path:
    if (package_root / "packages").is_symlink():
        raise ValueError("package payload root must not be a symlink")
    vendors = package_root / "packages" / "vendors"
    if vendors.is_symlink() or not vendors.is_dir():
        raise ValueError("package has no isolated experimental vendors directory")
    vendor = vendors / VENDOR
    if vendor.is_symlink() or not vendor.is_dir():
        raise ValueError(f"package is missing {VENDOR}")
    if {entry.name for entry in vendors.iterdir()} != {VENDOR}:
        raise ValueError("experimental package contains an unexpected vendor or config")
    for entry in vendor.rglob("*"):
        if entry.is_symlink():
            # Absolute links would still point into the extracted archive after
            # moving the staged tree and may dangle when makeself removes it.
            if (Path(os.readlink(entry)).is_absolute() or not entry.exists()
                    or not _within(entry.resolve(), vendor.resolve())):
                raise ValueError(f"package symlink escapes its vendor or is dangling: {entry}")
        elif not (entry.is_file() or entry.is_dir()):
            raise ValueError(f"unsupported package file type: {entry}")
    manifest = json.loads((vendor / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("mode") != "experimental" or manifest.get("vendor") != VENDOR:
        raise ValueError(f"package manifest must declare mode=experimental and vendor={VENDOR}")
    spec = importlib.util.spec_from_file_location("experimental_op_abi", package_root / "experimental_op_abi.py")
    abi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abi)
    abi.validated_contracts(vendor, require_manifest=True)
    return vendor


def _write_environment(vendor: Path) -> None:
    bin_dir = vendor / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "set_env.bash"
    # Never follow an otherwise valid payload link when generating this file.
    if script.is_symlink():
        script.unlink()
    script.write_text(
        '#!/bin/bash\n'
        '# Source in a fresh shell before starting the experimental Python process.\n'
        '_fla_exp_vendor="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"\n'
        'export ASCEND_CUSTOM_OPP_PATH="${_fla_exp_vendor}"\n'
        'export FLA_NPU_EXPERIMENTAL_OPP_PATH="$(cd -- "${_fla_exp_vendor}/../.." && pwd -P)"\n'
        'export FLA_NPU_DISABLE_PTH=1\n'
        'if [[ -f "${FLA_NPU_EXPERIMENTAL_OPP_PATH}/../layer_manifest.json" ]]; then\n'
        '    _fla_exp_env="$(python3 "${FLA_NPU_EXPERIMENTAL_OPP_PATH}/../../fla_npu_experimental_manage.py" '
        '--package-dir "${FLA_NPU_EXPERIMENTAL_OPP_PATH}/.." --shell-env)" || return 1\n'
        '    eval "${_fla_exp_env}"\n'
        '    unset _fla_exp_env\n'
        'fi\n'
        'unset _fla_exp_vendor\n',
        encoding="utf-8",
    )


def install(package_root: Path, raw_path: str) -> Path:
    destination = _validate_destination(raw_path)
    source = _validate_payload(package_root)
    # Build the complete tree beside the destination. POSIX rename can replace
    # an empty directory but cannot overwrite a directory that acquired files.
    with tempfile.TemporaryDirectory(prefix=".fla-experimental-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "opp"
        vendor = staged / "vendors" / VENDOR
        shutil.copytree(source, vendor, symlinks=True)
        _write_environment(vendor)
        (staged / "vendors" / "config.ini").write_text(f"load_priority={VENDOR}\n", encoding="utf-8")
        _validate_destination(raw_path)
        staged.rename(destination)
    return destination


def _without_makeself_context(argv: list[str]) -> list[str]:
    """Consume exactly the legacy makeself archive/cwd prefix, when verified.

    The bundled makeself header exports USER_PWD and prepends ``--$0`` and
    ``--$PWD`` before the user's arguments. These are metadata, not install
    options. Malformed or unverified prefixes remain for argparse to reject.
    """
    if len(argv) < 2 or not all(value.startswith("--") for value in argv[:2]):
        return argv
    archive = Path(argv[0][2:])
    invocation_dir = Path(argv[1][2:])
    # The self-extracting archive can legitimately be renamed for distribution.
    # Its identity comes from the payload manifest, not the outer filename.
    if archive.suffix != ".run":
        return argv
    user_pwd = os.environ.get("USER_PWD")
    if not user_pwd or not invocation_dir.is_absolute() or not invocation_dir.is_dir():
        return argv
    if Path(user_pwd).resolve() != invocation_dir.resolve():
        return argv
    if not archive.is_absolute():
        archive = invocation_dir / archive
    if not archive.is_file():
        return argv
    return argv[2:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--install-path", required=True, help="absolute isolated empty directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(_without_makeself_context(list(sys.argv[1:] if argv is None else argv)))
    try:
        destination = install(Path(__file__).resolve().parent, args.install_path)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"[experimental install] ERROR: {error}\n")
    if not args.quiet:
        print(f"[experimental install] Installed in {destination}")
        print("source " + shlex.quote(str(destination / "vendors" / VENDOR / "bin" / "set_env.bash")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
