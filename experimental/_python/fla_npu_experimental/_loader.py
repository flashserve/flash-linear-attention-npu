"""Load one explicitly selected experimental OPP in a fresh Python process.

This module never discovers stable wheels, registers torch operators, modifies
OPP search paths, or imports torch/torch_npu. Source the experimental vendor's
set_env.bash before starting Python, then import this package before torch_npu.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys


VENDOR = "fla_npu_experimental_transformer"
MANIFEST = "experimental-build-manifest.json"
_CONFIGURATION = None
_CONTENTS = None


def _absolute_path(value: str, label: str) -> Path:
    path = Path(value)
    if not value or not path.is_absolute():
        raise RuntimeError(f"{label} must be an explicit absolute path.")
    return path.resolve(strict=True)


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the experimental OPP root: {path}") from exc
    return resolved


def _resolve_configuration():
    if os.environ.get("FLA_NPU_DISABLE_PTH", "").upper() not in {"1", "TRUE", "YES", "ON"}:
        raise RuntimeError("Set FLA_NPU_DISABLE_PTH=1 before starting the experimental Python process.")
    bundled = Path(__file__).resolve().parent / "opp"
    root = _absolute_path(os.environ.get("FLA_NPU_EXPERIMENTAL_OPP_PATH", str(bundled) if bundled.is_dir() else ""),
                          "FLA_NPU_EXPERIMENTAL_OPP_PATH")
    if bundled.is_dir() and root != bundled.resolve():
        raise RuntimeError("Experimental wheel must use its own matching OPP; rebuild the wheel to update it.")
    vendor = _inside(root / "vendors" / VENDOR, root, "experimental vendor")
    manifest_path = _inside(vendor / MANIFEST, vendor, "experimental manifest")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("mode") != "experimental" or manifest.get("vendor") != VENDOR:
        raise RuntimeError("Experimental manifest mode/vendor does not match the selected OPP.")
    operators = manifest.get("operators")
    if not isinstance(operators, list):
        raise RuntimeError("Experimental manifest must contain an operators list.")
    if not operators or len({entry.get("name") for entry in operators}) != len(operators):
        raise RuntimeError("Experimental manifest must contain unique, nonempty operators.")
    for entry in operators:
        base = "aclnn" + entry.get("op_type", "")
        if not entry.get("name") or set(entry.get("aclnn_symbols", [])) != {base, base + "GetWorkspaceSize"}:
            raise RuntimeError("Experimental manifest has invalid aclnn entry points.")
    custom = _inside(vendor / "op_api" / "lib" / "libcust_opapi.so", vendor,
                     "experimental libcust_opapi.so")
    if not custom.is_file():
        raise RuntimeError(f"Experimental op_api library is not a file: {custom}")

    paths = [part for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if part]
    if not paths or any(_absolute_path(part, "ASCEND_CUSTOM_OPP_PATH") != vendor for part in paths):
        raise RuntimeError("ASCEND_CUSTOM_OPP_PATH must contain only the selected experimental vendor.")
    return root, vendor, custom


def _mapped_libraries():
    # Linux CANN can already be loaded through a launcher without a Python
    # module. Check the actual process mappings as well as sys.modules.
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return set()
    result = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            result.add(Path(fields[5].removesuffix(" (deleted)")).resolve())
    return result


def _check_process(custom: Path, *, first_import: bool):
    if any(name == "fla_npu" or name.startswith("fla_npu.") for name in sys.modules):
        raise RuntimeError("Stable fla_npu and fla_npu_experimental require separate Python processes.")
    if first_import and any(name == "torch_npu" or name.startswith("torch_npu.") or name == "acl"
                            for name in sys.modules):
        raise RuntimeError("Import fla_npu_experimental before torch_npu/acl in a fresh Python process.")
    for path in _mapped_libraries():
        if path.name == "libfla_npu_stable.so":
            raise RuntimeError("Stable launcher is already loaded; start a separate process.")
        if path.name == "libcust_opapi.so" and path != custom:
            raise RuntimeError(f"A different custom OPP is already loaded: {path}")
        if first_import and path.name in {"libascendcl.so", "libtorch_npu.so", "libopapi.so"}:
            raise RuntimeError("CANN is already loaded; import fla_npu_experimental in a fresh process first.")


class _RejectStableImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fla_npu" or fullname.startswith("fla_npu."):
            raise ImportError("Stable fla_npu cannot be imported after fla_npu_experimental; start a separate process.")
        return None


def configure():
    """Validate the selected installation without loading a shared library."""
    global _CONFIGURATION, _CONTENTS
    if _CONFIGURATION is None:
        configuration = _resolve_configuration()
        _check_process(configuration[2], first_import=True)
        _CONFIGURATION = configuration
        _CONTENTS = _contents(configuration[1])
        sys.meta_path.insert(0, _RejectStableImport())
    else:
        validate_active_environment()


def _contents(vendor):
    # Detect replacement in place after initialization without hashing host ELF
    # bytes on every operator call. The full ABI hash check runs before loading.
    files = [vendor / MANIFEST, vendor / "abi_manifest.json",
             vendor / "op_api/lib/libcust_opapi.so",
             *sorted((vendor / "op_api/include/aclnnop").glob("aclnn_*.h"))]
    return tuple((str(p), p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                 for p in files if p.exists())


def validate_active_environment():
    if _CONFIGURATION is None:
        raise RuntimeError("Experimental runtime has not been configured.")
    if _resolve_configuration() != _CONFIGURATION:
        raise RuntimeError("Experimental OPP selection changed after import; start a fresh process.")
    if _contents(_CONFIGURATION[1]) != _CONTENTS:
        raise RuntimeError("Experimental OPP contents changed after import; start a fresh process.")
    _check_process(_CONFIGURATION[2], first_import=False)


