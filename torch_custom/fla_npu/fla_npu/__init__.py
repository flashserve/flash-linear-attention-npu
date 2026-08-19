from __future__ import annotations

import ctypes
import os
import pathlib
from typing import Optional


_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
_DEFAULT_VENDOR_DIR = "fla_npu_transformer"
_ASCENDC_OPAPI_LIBRARIES: Optional[list[ctypes.CDLL]] = None


def _prepend_env_path(name: str, value: pathlib.Path) -> None:
    value_str = str(value)
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value_str not in parts:
        os.environ[name] = os.pathsep.join([value_str, *parts])


def _resolve_vendor_dir() -> pathlib.Path:
    package_dir = _PACKAGE_DIR.resolve()
    vendor_dir = package_dir / "opp" / "vendors" / _DEFAULT_VENDOR_DIR
    if vendor_dir.is_dir():
        resolved_vendor_dir = vendor_dir.resolve()
        try:
            resolved_vendor_dir.relative_to(package_dir)
        except ValueError:
            pass
        else:
            return resolved_vendor_dir

    raise FileNotFoundError(
        "Unable to find the FLA NPU custom OPP embedded in this Python package. "
        f"Expected a regular package-local OPP at {vendor_dir}. Reinstall the "
        "complete wheel. External CANN vendor directories are not used as a "
        "runtime library fallback."
    )


def _resolve_packaged_opapi(vendor_dir: pathlib.Path) -> pathlib.Path:
    package_dir = _PACKAGE_DIR.resolve()
    op_api_lib = (vendor_dir / "op_api" / "lib" / "libcust_opapi.so").resolve()
    try:
        op_api_lib.relative_to(package_dir)
    except ValueError:
        pass
    else:
        if op_api_lib.is_file():
            return op_api_lib
    raise FileNotFoundError(
        "Unable to find the packaged FLA NPU op_api library. Expected "
        f"{vendor_dir / 'op_api' / 'lib' / 'libcust_opapi.so'}. Reinstall the "
        "complete wheel."
    )


def _prepare_embedded_opp() -> pathlib.Path:
    if not (os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_OPP_PATH")):
        raise RuntimeError(
            "CANN environment is not initialized. Please source the CANN set_env.sh "
            "before importing fla_npu."
        )

    vendor_dir = _resolve_vendor_dir()
    op_api_lib = _resolve_packaged_opapi(vendor_dir)
    op_api_alias = op_api_lib.with_name("libopapi.so")
    if op_api_alias.exists() or op_api_alias.is_symlink():
        raise RuntimeError(
            "The FLA NPU custom OPP contains libopapi.so, which can shadow the "
            "CANN runtime library. Reinstall a wheel that only contains "
            f"libcust_opapi.so, or remove the stale alias: {op_api_alias}"
        )

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir.parent.parent)
    os.environ["FLA_NPU_OP_API_LIB"] = str(op_api_lib)
    return vendor_dir


def _load_shared_library_required(path_or_name) -> ctypes.CDLL:
    mode = (
        getattr(os, "RTLD_LOCAL", 0)
        | getattr(os, "RTLD_NOW", 0)
        | getattr(os, "RTLD_NODELETE", 0)
    )
    return ctypes.CDLL(str(path_or_name), mode=mode)


def load_ascendc_opapi_libraries() -> list[ctypes.CDLL]:
    """Load CANN and packaged FLA opapi without publishing global symbols."""

    global _ASCENDC_OPAPI_LIBRARIES
    if _ASCENDC_OPAPI_LIBRARIES is not None:
        return _ASCENDC_OPAPI_LIBRARIES

    vendor_dir = _prepare_embedded_opp()
    custom_opapi = _resolve_packaged_opapi(vendor_dir)

    try:
        cann_library = _load_shared_library_required("libopapi.so")
    except OSError as exc:
        raise RuntimeError(
            "Unable to load the CANN op_api library libopapi.so. Please source "
            "the matching CANN set_env.sh before importing fla_npu. "
            f"Dynamic loader error: {exc}"
        ) from exc

    try:
        custom_library = _load_shared_library_required(custom_opapi)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load packaged FLA NPU custom op_api library: {custom_opapi}. "
            f"Dynamic loader error: {exc}"
        ) from exc

    # CANN is loaded first, while the explicit custom handle remains available
    # to the legacy C++ extension through FLA_NPU_OP_API_LIB.
    _ASCENDC_OPAPI_LIBRARIES = [custom_library, cann_library]
    return _ASCENDC_OPAPI_LIBRARIES


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


# Load the custom operator library
def _load_opextension_so():
    load_ascendc_opapi_libraries()

    import torch
    import torch_npu

    _preload_torch_npu_dependencies(torch, torch_npu)

    so_dir = _PACKAGE_DIR
    so_files = list(so_dir.glob('custom_aclnn_extension_lib*.so'))

    if not so_files:
        raise FileNotFoundError(f"not find custom_aclnn_extension_lib*.so in {so_dir}")

    atb_so_path = str(so_files[0])
    torch.ops.load_library(atb_so_path)
    from .ops.ascendc import install_torch_npu_ops_compat

    install_torch_npu_ops_compat()

_load_opextension_so()
