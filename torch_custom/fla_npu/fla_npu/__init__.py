from __future__ import annotations

import ctypes
import importlib
import os
import pathlib


_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
_DEFAULT_VENDOR_DIR = "fla_npu_transformer"
_ASCENDC_EXTENSION_MODULE = None
_ASCENDC_EXTENSION_LIBRARY: pathlib.Path | None = None
_LEGACY_TORCH_OPS_LOADED = False
_LEGACY_TORCH_OPS_LIBRARY: pathlib.Path | None = None


def _prepend_env_path(name: str, value: pathlib.Path) -> None:
    value_str = str(value)
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value_str not in parts:
        os.environ[name] = os.pathsep.join([value_str, *parts])


def _candidate_vendor_names() -> list[str]:
    return [_DEFAULT_VENDOR_DIR]


def _candidate_opp_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    override = os.environ.get("FLA_NPU_OPP_PATH", "").strip()
    if override:
        roots.append(pathlib.Path(override).expanduser())

    roots.append(_PACKAGE_DIR / "opp")

    for env_name in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH"):
        for part in os.environ.get(env_name, "").split(os.pathsep):
            if part:
                roots.append(pathlib.Path(part).expanduser())

    return list(dict.fromkeys(roots))


def _resolve_vendor_dir() -> pathlib.Path:
    for root in _candidate_opp_roots():
        if (root / "op_api" / "lib" / "libcust_opapi.so").exists():
            return root.resolve()

        vendors_root = root / "vendors"
        for name in _candidate_vendor_names():
            vendor_dir = vendors_root / name
            if vendor_dir.exists():
                return vendor_dir.resolve()

        if vendors_root.exists():
            vendor_dirs = [path for path in vendors_root.iterdir() if path.is_dir()]
            if len(vendor_dirs) == 1:
                return vendor_dirs[0].resolve()

    raise FileNotFoundError(
        "Unable to find FLA NPU custom OPP. Expected "
        f"{_PACKAGE_DIR / 'opp' / 'vendors' / _DEFAULT_VENDOR_DIR}. "
        "If you installed it separately, source the custom OPP set_env.bash, "
        "or set FLA_NPU_OPP_PATH to either the OPP root containing vendors/ "
        "or the vendor directory itself."
    )


def _prepare_embedded_opp() -> pathlib.Path:
    if not (os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_OPP_PATH")):
        raise RuntimeError(
            "CANN environment is not initialized. Please source the CANN set_env.sh "
            "before calling fla_npu Ascend C operators."
        )

    vendor_dir = _resolve_vendor_dir()
    opp_root = vendor_dir.parent.parent if vendor_dir.parent.name == "vendors" else vendor_dir
    op_api_lib = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    if not op_api_lib.exists():
        raise FileNotFoundError(f"Embedded custom op_api library not found: {op_api_lib}")
    op_api_alias = op_api_lib.with_name("libopapi.so")

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", opp_root)
    _prepend_env_path("LD_LIBRARY_PATH", op_api_lib.parent)
    os.environ["FLA_NPU_OP_API_LIB"] = str(op_api_lib)

    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    if op_api_alias.exists():
        ctypes.CDLL(str(op_api_alias), mode=mode)
    ctypes.CDLL(str(op_api_lib), mode=mode)
    return vendor_dir


def _find_ascendc_extension_library() -> pathlib.Path:
    so_files = list(_PACKAGE_DIR.glob("custom_aclnn_extension_lib*.so"))
    if not so_files:
        raise FileNotFoundError(f"not find custom_aclnn_extension_lib*.so in {_PACKAGE_DIR}")
    return so_files[0].resolve()


def _preload_library(path: pathlib.Path) -> None:
    if not path.exists():
        return
    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    ctypes.CDLL(str(path), mode=mode)


def _preload_torch_npu_dependencies(torch_module, torch_npu_module) -> None:
    torch_lib = pathlib.Path(torch_module.__file__).resolve().parent / "lib"
    torch_npu_lib = pathlib.Path(torch_npu_module.__file__).resolve().parent / "lib"
    _prepend_env_path("LD_LIBRARY_PATH", torch_lib)
    _prepend_env_path("LD_LIBRARY_PATH", torch_npu_lib)

    for lib_path in (
        torch_lib / "libc10.so",
        torch_lib / "libtorch.so",
        torch_lib / "libtorch_cpu.so",
        torch_npu_lib / "libtorch_npu.so",
    ):
        _preload_library(lib_path)


def load_ascendc_extension():
    """Load the direct Ascend C Python extension without torch.ops dispatcher."""

    global _ASCENDC_EXTENSION_MODULE, _ASCENDC_EXTENSION_LIBRARY
    if _ASCENDC_EXTENSION_MODULE is not None:
        return _ASCENDC_EXTENSION_MODULE

    _prepare_embedded_opp()

    import torch
    import torch_npu

    _preload_torch_npu_dependencies(torch, torch_npu)
    _ASCENDC_EXTENSION_LIBRARY = _find_ascendc_extension_library()
    _ASCENDC_EXTENSION_MODULE = importlib.import_module("fla_npu.custom_aclnn_extension_lib")
    return _ASCENDC_EXTENSION_MODULE


def load_legacy_torch_ops() -> pathlib.Path:
    """Load the legacy PyTorch dispatcher custom ops.

    ``import fla_npu`` is intentionally lightweight and does not import torch,
    torch_npu, or register ``torch.ops.npu`` kernels.  Call this function only
    when old call sites such as ``torch.ops.npu.npu_chunk_fwd_o(...)`` must keep
    working during the migration to the decoupled ``fla_npu.ops.ascendc`` API.
    """

    global _LEGACY_TORCH_OPS_LOADED, _LEGACY_TORCH_OPS_LIBRARY
    if _LEGACY_TORCH_OPS_LOADED and _LEGACY_TORCH_OPS_LIBRARY is not None:
        return _LEGACY_TORCH_OPS_LIBRARY

    import torch

    load_ascendc_extension()
    legacy_library = _ASCENDC_EXTENSION_LIBRARY or _find_ascendc_extension_library()
    torch.ops.load_library(str(legacy_library))
    from .ops.ascendc import install_legacy_torch_ops_warning, install_torch_npu_ops_compat

    install_torch_npu_ops_compat()
    install_legacy_torch_ops_warning()
    _LEGACY_TORCH_OPS_LOADED = True
    _LEGACY_TORCH_OPS_LIBRARY = legacy_library
    return legacy_library


def is_legacy_torch_ops_loaded() -> bool:
    return _LEGACY_TORCH_OPS_LOADED


__all__ = ["is_legacy_torch_ops_loaded", "load_ascendc_extension", "load_legacy_torch_ops"]
