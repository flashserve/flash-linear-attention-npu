from __future__ import annotations

import ctypes
import os
import pathlib
import warnings
from typing import Optional


_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
_DEFAULT_VENDOR_DIR = "fla_npu_transformer"
_ASCENDC_OPAPI_LIBRARIES: Optional[list[ctypes.CDLL]] = None
_LEGACY_TORCH_OPS_LOADED = False
_LEGACY_TORCH_OPS_LIBRARY: Optional[pathlib.Path] = None


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
        "complete wheel, or overlay the "
        "matching run package into the installed fla_npu package. External CANN "
        "vendor directories are not used as a fallback."
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
        "complete wheel, or overlay the matching run package into the installed "
        "fla_npu package."
    )


def _prepare_embedded_opp() -> pathlib.Path:
    if not (os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_OPP_PATH")):
        raise RuntimeError(
            "CANN environment is not initialized. Please source the CANN set_env.sh "
            "before importing fla_npu."
        )

    vendor_dir = _resolve_vendor_dir()
    public_op_api_dir = vendor_dir / "op_api" / "lib"
    op_api_lib = _resolve_packaged_opapi(vendor_dir)
    op_api_alias = public_op_api_dir / "libopapi.so"
    if op_api_alias.exists() or op_api_alias.is_symlink():
        try:
            op_api_alias.unlink()
        except OSError as exc:
            raise RuntimeError(
                "The FLA NPU custom OPP contains a stale libopapi.so that can "
                "shadow the CANN runtime library, and automatic cleanup failed. "
                f"Remove the stale alias and retry: {op_api_alias}"
            ) from exc
        warnings.warn(
            "Removed a stale FLA NPU libopapi.so alias left by an older package. "
            "CANN must provide libopapi.so; the custom package only provides "
            "libcust_opapi.so.",
            RuntimeWarning,
            stacklevel=2,
        )

    # CANN still needs the package OPP for host/tiling/kernel discovery. Do not
    # add op_api/lib to LD_LIBRARY_PATH: FLA opens this exact package-local file
    # by absolute path, while other RTLD_LOCAL consumers keep their own search.
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir.parent.parent)
    os.environ["FLA_NPU_OP_API_LIB"] = str(op_api_lib)

    return vendor_dir


def _warn_if_embedded_opp_not_preconfigured() -> None:
    """Warn when the wheel-embedded custom OPP was not pre-configured.

    CANN discovers custom operator host/tiling/kernel binaries when the runtime
    is initialized (e.g. at ``import torch_npu``). If CANN initializes before
    ``import fla_npu``, setting ``ASCEND_CUSTOM_OPP_PATH`` in this process may be
    too late for the already-initialized kernel registry (issue #429).
    """
    try:
        vendor_dir = _resolve_vendor_dir()
    except Exception:
        return
    parts = [part for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if part]
    if str(vendor_dir) in parts:
        return
    warnings.warn(
        "[fla-npu] ASCEND_CUSTOM_OPP_PATH does not contain the custom OPP implements "
        "before import fla_npu, If you need to use fla_npu operators,  run:\n\n"
        f"  export ASCEND_CUSTOM_OPP_PATH=\"{vendor_dir}:{vendor_dir / 'op_api' / 'lib'}:${{ASCEND_CUSTOM_OPP_PATH:-}}\"",
        RuntimeWarning,
        stacklevel=2,
    )


def _load_shared_library_required(path_or_name) -> ctypes.CDLL:
    mode = (
        getattr(os, "RTLD_LOCAL", 0)
        | getattr(os, "RTLD_NOW", 0)
        | getattr(os, "RTLD_NODELETE", 0)
    )
    return ctypes.CDLL(str(path_or_name), mode=mode)


def load_ascendc_opapi_libraries() -> list[ctypes.CDLL]:
    """Load CANN and custom op_api libraries for custom-first symbol lookup."""

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
            f"Unable to load embedded FLA NPU custom op_api library: {custom_opapi}. "
            f"Dynamic loader error: {exc}"
        ) from exc
    # Load CANN first so its runtime is initialized independently, but search
    # the explicit handles custom-first when an aclnn symbol exists in both.
    libraries = [custom_library, cann_library]

    _ASCENDC_OPAPI_LIBRARIES = libraries
    return libraries


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
    """Backward-compatible alias for loading the decoupled Ascend C runtime."""

    return load_ascendc_opapi_libraries()


def load_legacy_torch_ops() -> pathlib.Path:
    """Load the legacy PyTorch dispatcher custom ops.

    ``import fla_npu`` initializes the decoupled Ascend C runtime, but does not
    import torch, torch_npu, or register ``torch.ops.npu`` kernels. Call this
    function only when old call sites such as
    ``torch.ops.npu.npu_chunk_fwd_o(...)`` must keep working during the
    migration to the decoupled ``fla_npu.ops.ascendc`` API.
    """

    global _LEGACY_TORCH_OPS_LOADED, _LEGACY_TORCH_OPS_LIBRARY
    if _LEGACY_TORCH_OPS_LOADED and _LEGACY_TORCH_OPS_LIBRARY is not None:
        return _LEGACY_TORCH_OPS_LIBRARY

    import torch
    import torch_npu

    load_ascendc_opapi_libraries()
    _preload_torch_npu_dependencies(torch, torch_npu)
    legacy_library = _find_ascendc_extension_library()
    torch.ops.load_library(str(legacy_library))
    from .ops.ascendc import install_legacy_torch_ops_warning, install_torch_npu_ops_compat

    install_torch_npu_ops_compat()
    install_legacy_torch_ops_warning()
    _LEGACY_TORCH_OPS_LOADED = True
    _LEGACY_TORCH_OPS_LIBRARY = legacy_library
    return legacy_library


def is_legacy_torch_ops_loaded() -> bool:
    return _LEGACY_TORCH_OPS_LOADED


__all__ = [
    "is_legacy_torch_ops_loaded",
    "load_ascendc_extension",
    "load_ascendc_opapi_libraries",
    "load_legacy_torch_ops",
]


def _run_import_guards() -> None:
    """Run tier/version guards before the embedded OPP is loaded."""
    try:
        from . import _guard as _fla_guard

        _fla_guard.run_guards()
    except RuntimeError:
        raise
    except Exception as exc:  # guard defects must never break import
        warnings.warn(f"fla_npu environment guard skipped: {exc}", RuntimeWarning)


_run_import_guards()
_warn_if_embedded_opp_not_preconfigured()
load_ascendc_opapi_libraries()
