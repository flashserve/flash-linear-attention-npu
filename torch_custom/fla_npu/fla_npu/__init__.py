from __future__ import annotations

import ctypes
import importlib.util
import os
import pathlib
import re
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
_OVERLAY_VENDOR_NAME_RE = re.compile(
    rf"{re.escape(_OVERLAY_VENDOR_PREFIX)}[A-Za-z0-9][A-Za-z0-9_.-]*\Z"
)
_ASCENDC_OPAPI_LIBRARIES: Optional[list[ctypes.CDLL]] = None


def _prepend_env_path(name: str, value: pathlib.Path) -> None:
    value_str = str(value)
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value_str not in parts:
        os.environ[name] = os.pathsep.join([value_str, *parts])


def _is_package_opp_path(value: str) -> bool:
    try:
        pathlib.Path(value).resolve().relative_to((_PACKAGE_DIR / "opp").resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _normalize_custom_opp_path(vendor_dirs: list[pathlib.Path]) -> None:
    """Put package vendors first and discard obsolete package-local entries."""

    current = [
        part
        for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if part
    ]
    package_entries = [part for part in current if _is_package_opp_path(part)]
    external_entries = [part for part in current if not _is_package_opp_path(part)]
    normalized = [str(path) for path in vendor_dirs] + external_entries
    if package_entries and current != normalized:
        warnings.warn(
            "fla_npu: normalized package-local ASCEND_CUSTOM_OPP_PATH entries "
            "to overlay-first vendor order and removed legacy OPP-root/op_api-lib entries.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(normalized)


def overlay_vendor_names() -> list[str]:
    """Names of the overlay vendors, highest priority first.

    ``FLA_NPU_OVERLAY_VENDOR`` (``os.pathsep``-separated) wins when it is set.
    Otherwise the overlays recorded in ``opp/vendors/config.ini`` are used, so
    that ``install_opp --overlay --overlay-name <name>`` takes effect without
    any extra environment variable.
    """

    raw = os.environ.get(_OVERLAY_VENDOR_ENV)
    if raw is not None:
        names: list[str] = []
        for candidate in raw.split(os.pathsep):
            name = candidate.strip()
            if not name:
                continue
            if not _OVERLAY_VENDOR_NAME_RE.fullmatch(name):
                warnings.warn(
                    f"fla_npu: ignoring invalid {_OVERLAY_VENDOR_ENV} entry {name!r}; "
                    f"overlay names must match {_OVERLAY_VENDOR_PREFIX}<name>.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if name not in names:
                names.append(name)
        return names
    return _discover_overlay_vendor_names()


def _validate_overlay_abi(vendor_dirs: list[pathlib.Path]) -> None:
    """Fail before loading code when an overlay changes the wheel's aclnn ABI."""

    try:
        from ._op_abi import compare_overlay_to_base
    except ImportError:  # run_path/direct-script diagnostics lack package context
        abi_path = _PACKAGE_DIR / "_op_abi.py"
        abi_spec = importlib.util.spec_from_file_location(
            "fla_npu_standalone_op_abi", abi_path
        )
        if abi_spec is None or abi_spec.loader is None:
            raise
        abi_module = importlib.util.module_from_spec(abi_spec)
        abi_spec.loader.exec_module(abi_module)
        compare_overlay_to_base = abi_module.compare_overlay_to_base

    base_vendor = next(
        (path for path in vendor_dirs if path.name == _DEFAULT_VENDOR_DIR), None
    )
    if base_vendor is None:
        raise RuntimeError(
            "fla_npu: no base vendor is available to validate overlay aclnn ABI. "
            "Reinstall the complete wheel before using an overlay."
        )
    failures: list[str] = []
    for vendor_dir in vendor_dirs:
        if vendor_dir == base_vendor:
            continue
        failures.extend(
            f"{vendor_dir.name}: {problem}"
            for problem in compare_overlay_to_base(vendor_dir, base_vendor)
        )
    if failures:
        detail = "\n  - ".join(failures)
        raise RuntimeError(
            "fla_npu: overlay aclnn ABI differs from the installed wheel. The "
            "Stable launcher and wrappers cannot safely call this overlay; "
            "rebuild a matching overlay or wheel. For recovery without loading "
            "vendor code, run python -m fla_npu_opp_manage --status or "
            "python -m fla_npu_opp_manage --remove-overlay --overlay-name <name>."
            f"\n  - {detail}"
        )


def _discover_overlay_vendor_names() -> list[str]:
    """Overlay vendors present in the package, in ``config.ini`` priority order."""

    vendors_root = _PACKAGE_DIR.resolve() / "opp" / "vendors"
    names: list[str] = []

    def _add(candidate: str) -> None:
        name = candidate.strip()
        if not name or name == _DEFAULT_VENDOR_DIR or name in names:
            return
        if not _OVERLAY_VENDOR_NAME_RE.fullmatch(name):
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
    vendors_root = package_dir / "opp" / "vendors"
    if name != _DEFAULT_VENDOR_DIR and not _OVERLAY_VENDOR_NAME_RE.fullmatch(name):
        return None
    vendor_dir = vendors_root / name
    if not vendor_dir.is_dir():
        return None
    resolved_vendor_dir = vendor_dir.resolve()
    try:
        resolved_vendor_dir.relative_to(vendors_root)
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
    _validate_overlay_abi(vendor_dirs)
    op_api_libs: list[pathlib.Path] = []
    for vendor_dir in vendor_dirs:
        op_api_libs.append(_resolve_packaged_opapi(vendor_dir))

    for vendor_dir in vendor_dirs:
        _drop_stale_libopapi_alias(vendor_dir)

    # ASCEND_CUSTOM_OPP_PATH entries are vendor directories. Rebuild all entries
    # owned by this package so shell source order cannot put the base before an
    # overlay; keep unrelated external entries behind the package vendors.
    _normalize_custom_opp_path(vendor_dirs)

    # Retain the complete base path for single-vendor integrations. Stable
    # execution uses the ordered plural list whenever overlays are enabled.
    base_vendor = next(
        (vendor_dir for vendor_dir in vendor_dirs if vendor_dir.name == _DEFAULT_VENDOR_DIR),
        vendor_dirs[0],
    )
    os.environ["FLA_NPU_OP_API_LIB"] = str(_resolve_packaged_opapi(base_vendor))
    os.environ["FLA_NPU_OP_API_LIBS"] = os.pathsep.join(str(path) for path in op_api_libs)
    # The actual launcher (including FLA_NPU_STABLE_LIB overrides) is checked
    # in _stable.load(), after its stamp and before any op is resolved. Do not
    # import torch here: OPP paths must be ready before CANN initializes.
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
        f"  export ASCEND_CUSTOM_OPP_PATH=\"{vendor_dir}:${{ASCEND_CUSTOM_OPP_PATH:-}}\"",
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


def load_ascendc_extension():
    """Backward-compatible alias for loading the decoupled Ascend C runtime."""

    return load_ascendc_opapi_libraries()


__all__ = ["overlay_vendor_names", "load_ascendc_extension", "load_ascendc_opapi_libraries"]


def _run_import_guards() -> None:
    """Run tier/version guards before the embedded OPP is loaded."""
    try:
        from . import _guard as _fla_guard

        _fla_guard.run_guards()
    except RuntimeError:
        raise
    except Exception as exc:  # guard defects must never break import
        warnings.warn(f"fla_npu environment guard skipped: {exc}", RuntimeWarning)


from ._backend_policy import validate_backend_environment as _validate_backend_environment

_validate_backend_environment()
_run_import_guards()
_warn_if_embedded_opp_not_preconfigured()
load_ascendc_opapi_libraries()
