from __future__ import annotations

import ctypes
import os
import pathlib
import warnings
from typing import Optional


_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
_DEFAULT_VENDOR_DIR = "fla_npu_transformer"
# Overlay vendors live next to the base vendor inside the installed
# package and take priority over it. FLA_NPU_OVERLAY_VENDOR accepts an
# os.pathsep-separated list; the first entry wins.
_DEFAULT_OVERLAY_VENDOR_DIR = "fla_npu_overlay_transformer"
# Any package-local vendor directory with this prefix is an overlay: it outranks
# the base vendor and is auto-discovered when FLA_NPU_OVERLAY_VENDOR is unset.
_OVERLAY_VENDOR_PREFIX = "fla_npu_overlay_"
_OVERLAY_VENDOR_ENV = "FLA_NPU_OVERLAY_VENDOR"
_STABLE_ABI_ENV = "FLA_NPU_STABLE_ABI"
_ASCENDC_OPAPI_LIBRARIES: Optional[list[ctypes.CDLL]] = None
_LEGACY_TORCH_OPS_LOADED = False
_LEGACY_TORCH_OPS_LIBRARY: Optional[pathlib.Path] = None


def _prepend_env_path(name: str, value: pathlib.Path) -> None:
    value_str = str(value)
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value_str not in parts:
        os.environ[name] = os.pathsep.join([value_str, *parts])


def overlay_vendor_names() -> list[str]:
    """Names of the overlay vendors, highest priority first.

    ``FLA_NPU_OVERLAY_VENDOR`` (``os.pathsep``-separated) wins when it is set.
    Otherwise the overlays recorded in ``opp/vendors/config.ini`` are used, so
    that ``install_opp --overlay --overlay-name <name>`` takes effect without
    any extra environment variable.
    """

    raw = os.environ.get(_OVERLAY_VENDOR_ENV)
    if raw is not None:
        names = [name.strip() for name in raw.split(os.pathsep)]
        return [name for name in names if name]
    return _discover_overlay_vendor_names()


def _launcher_supports_overlay_order() -> bool:
    """True when the bundled launcher honours the ordered ``FLA_NPU_OP_API_LIBS`` list.

    Older launchers resolve every aclnn symbol from the single library named by
    ``FLA_NPU_OP_API_LIB`` (``Runtime::symbol()`` only dlsym()s one custom
    handle), so with an overlay installed they would silently run the *base*
    implementation.  Detecting the capability lets us pick a correct backend
    instead of leaving the outcome to an unrelated environment variable.
    """

    try:
        return b"FLA_NPU_OP_API_LIBS" in (_PACKAGE_DIR / "libfla_npu_stable.so").read_bytes()
    except OSError:
        return False


def _apply_overlay_backend_policy(vendor_dirs: list[pathlib.Path]) -> None:
    """Keep "an overlay is installed" and "which implementation runs" consistent.

    * no overlay                     → nothing to do (default backend rules apply)
    * overlay + capable launcher     → launcher searches overlay first, so the
                                       launcher may stay the default backend
    * overlay + legacy launcher      → force the ctypes reference path (which
                                       resolves symbols per library); if the user
                                       explicitly asked for the launcher, warn
                                       loudly instead of failing silently
    """

    if len(vendor_dirs) < 2:
        return
    requested = (os.environ.get(_STABLE_ABI_ENV) or "").strip()
    if _launcher_supports_overlay_order():
        return
    if not requested:
        os.environ[_STABLE_ABI_ENV] = "ctypes"
        return
    if requested == "ctypes":
        return
    warnings.warn(
        "fla_npu: an overlay vendor is installed, but FLA_NPU_STABLE_ABI="
        f"{requested!r} selects a Stable-ABI launcher that resolves every aclnn "
        "symbol from the single library in FLA_NPU_OP_API_LIB. The overlaid "
        "operators would silently run the base implementation instead of the "
        "overlay one. Rebuild the launcher (python csrc/build_stable.py) to get "
        "ordered FLA_NPU_OP_API_LIBS lookup, or unset FLA_NPU_STABLE_ABI to use "
        "the ctypes reference path.",
        RuntimeWarning,
        stacklevel=2,
    )


def _discover_overlay_vendor_names() -> list[str]:
    """Overlay vendors present in the package, in ``config.ini`` priority order."""

    vendors_root = _PACKAGE_DIR.resolve() / "opp" / "vendors"
    names: list[str] = []

    def _add(candidate: str) -> None:
        name = candidate.strip()
        if not name or name == _DEFAULT_VENDOR_DIR or name in names:
            return
        if not name.startswith(_OVERLAY_VENDOR_PREFIX):
            return
        if _package_local_vendor_dir(name) is not None:
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


def _package_local_vendor_dir(name: str) -> Optional[pathlib.Path]:
    """Return ``<pkg>/opp/vendors/<name>`` when it exists inside the package."""

    package_dir = _PACKAGE_DIR.resolve()
    vendor_dir = package_dir / "opp" / "vendors" / name
    if not vendor_dir.is_dir():
        return None
    resolved_vendor_dir = vendor_dir.resolve()
    try:
        resolved_vendor_dir.relative_to(package_dir)
    except ValueError:
        return None
    return resolved_vendor_dir


def _resolve_vendor_dirs() -> list[pathlib.Path]:
    """Ordered vendor overlay list: overlay vendors first, base last."""

    vendor_dirs: list[pathlib.Path] = []
    for name in [*overlay_vendor_names(), _DEFAULT_VENDOR_DIR]:
        resolved_vendor_dir = _package_local_vendor_dir(name)
        if resolved_vendor_dir is not None and resolved_vendor_dir not in vendor_dirs:
            vendor_dirs.append(resolved_vendor_dir)
    if vendor_dirs:
        return vendor_dirs

    package_dir = _PACKAGE_DIR.resolve()
    raise FileNotFoundError(
        "Unable to find the FLA NPU custom OPP embedded in this Python package. "
        f"Expected a regular package-local OPP at {package_dir / 'opp' / 'vendors' / _DEFAULT_VENDOR_DIR} "
        "or an overlay vendor at "
        f"{package_dir / 'opp' / 'vendors' / _DEFAULT_OVERLAY_VENDOR_DIR}. Reinstall the "
        "complete wheel, or overlay the "
        "matching run package into the installed fla_npu package. External CANN "
        "vendor directories are not used as a fallback."
    )


def _resolve_vendor_dir() -> pathlib.Path:
    """Base vendor directory (kept for callers that expect a single vendor)."""

    package_dir = _PACKAGE_DIR.resolve()
    base_vendor_dir = _package_local_vendor_dir(_DEFAULT_VENDOR_DIR)
    if base_vendor_dir is not None:
        return base_vendor_dir
    # No base vendor: fall back to the highest-priority overlay vendor.
    return _resolve_vendor_dirs()[0]


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


def _drop_stale_libopapi_alias(vendor_dir: pathlib.Path) -> None:
    op_api_alias = vendor_dir / "op_api" / "lib" / "libopapi.so"
    if not (op_api_alias.exists() or op_api_alias.is_symlink()):
        return
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


def _prepare_embedded_opp() -> list[pathlib.Path]:
    if not (os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_OPP_PATH")):
        raise RuntimeError(
            "CANN environment is not initialized. Please source the CANN set_env.sh "
            "before importing fla_npu."
        )

    vendor_dirs = _resolve_vendor_dirs()
    op_api_libs: list[pathlib.Path] = []
    for vendor_dir in vendor_dirs:
        op_api_libs.append(_resolve_packaged_opapi(vendor_dir))

    # Prepend in reverse so the final ASCEND_CUSTOM_OPP_PATH lists the overlay
    # (overlay) vendors before the base vendor: kernel/tiling discovery follows
    # this order.
    for vendor_dir in reversed(vendor_dirs):
        _drop_stale_libopapi_alias(vendor_dir)
        # CANN still needs the package OPP for host/tiling/kernel discovery. Do not
        # add op_api/lib to LD_LIBRARY_PATH: FLA opens this exact package-local file
        # by absolute path, while other RTLD_LOCAL consumers keep their own search.
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir.parent.parent)

    # The first vendor is the highest-priority implementation; keep the legacy
    # single-library variable pointing at it and expose the full order too.
    # The Stable-ABI launcher (libfla_npu_stable.so) resolves every aclnn symbol
    # from the single library named here, so it must point at a *complete* op set:
    # an overlay vendor alone would hide every operator it does not carry.
    base_vendor = next(
        (vendor_dir for vendor_dir in vendor_dirs if vendor_dir.name == _DEFAULT_VENDOR_DIR),
        vendor_dirs[0],
    )
    os.environ["FLA_NPU_OP_API_LIB"] = str(_resolve_packaged_opapi(base_vendor))
    os.environ["FLA_NPU_OP_API_LIBS"] = os.pathsep.join(str(path) for path in op_api_libs)
    # The launcher's symbol lookup decides whether an overlay can be honoured:
    # legacy launchers resolve every symbol from the single FLA_NPU_OP_API_LIB
    # handle, plural-capable launchers search the ordered FLA_NPU_OP_API_LIBS
    # list (overlay first).  Keep the backend choice consistent with that.
    _apply_overlay_backend_policy(vendor_dirs)
    os.environ.setdefault("FLA_NPU_OPP_PATH", str(vendor_dirs[0].parent.parent))

    return vendor_dirs


def _warn_if_embedded_opp_not_preconfigured() -> None:
    """Warn when the wheel-embedded custom OPP was not pre-configured.

    CANN discovers custom operator host/tiling/kernel binaries when the runtime
    is initialized (e.g. at ``import torch_npu``). If CANN initializes before
    ``import fla_npu``, setting ``ASCEND_CUSTOM_OPP_PATH`` in this process may be
    too late for the already-initialized kernel registry (issue #429).
    """
    try:
        vendor_dirs = _resolve_vendor_dirs()
    except Exception:
        return
    parts = [part for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if part]
    missing = [vendor_dir for vendor_dir in vendor_dirs if str(vendor_dir) not in parts]
    if not missing:
        return
    vendor_dir = missing[0]
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
    """Load CANN and custom op_api libraries for custom-first symbol lookup.

    The returned list is searched in order, so overlay vendors shadow
    the base vendor, which in turn shadows CANN. Symbols a vendor does not
    provide fall through to the next library.
    """

    global _ASCENDC_OPAPI_LIBRARIES
    if _ASCENDC_OPAPI_LIBRARIES is not None:
        return _ASCENDC_OPAPI_LIBRARIES

    vendor_dirs = _prepare_embedded_opp()

    try:
        cann_library = _load_shared_library_required("libopapi.so")
    except OSError as exc:
        raise RuntimeError(
            "Unable to load the CANN op_api library libopapi.so. Please source "
            "the matching CANN set_env.sh before importing fla_npu. "
            f"Dynamic loader error: {exc}"
        ) from exc

    custom_libraries: list[ctypes.CDLL] = []
    for vendor_dir in vendor_dirs:
        custom_opapi = _resolve_packaged_opapi(vendor_dir)
        try:
            custom_libraries.append(_load_shared_library_required(custom_opapi))
        except OSError as exc:
            raise RuntimeError(
                f"Unable to load embedded FLA NPU custom op_api library: {custom_opapi}. "
                f"Dynamic loader error: {exc}"
            ) from exc
    # Load CANN first so its runtime is initialized independently, but search
    # the explicit handles custom-first when an aclnn symbol exists in both.
    # Overlay vendors come first, so a the overlay overrides the base vendor.
    libraries = [*custom_libraries, cann_library]

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
    "overlay_vendor_names",
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
