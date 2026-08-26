"""Shared capability probes for the optional legacy PyTorch extension build."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

from packaging.version import InvalidVersion, Version


TORCHNPUGEN_MODULES = (
    "torchnpugen.gen_op_plugin_functions",
    "torchnpugen.gen_derivatives",
    "torchnpugen.gen_op_backend",
    "torchnpugen.gen_backend_stubs",
    "torchnpugen.struct.gen_struct_opapi",
)

CPP_EXTENSION_MODULE = "torch.utils.cpp_extension"
CPP_EXTENSION_SYMBOLS = ("BuildExtension", "CppExtension")

KNOWN_GDN_STREAM_FIX_MINIMUMS = {
    "2.7.1": "2.7.1.post5",
    "2.8.0": "2.8.0.post5",
    "2.9.0": "2.9.0.post3",
    "2.10.0": "2.10.0.post2",
    "2.11.0": "2.11.0rc3",
    "2.12.0": "2.12.0rc1",
}
MIN_TORCH_NPU_FUTURE_STREAM_FIX_FAMILY = "2.13.0"
TORCH_NPU_GDN_STREAM_FIX_RELEASE_URL = (
    "https://gitcode.com/Ascend/pytorch/releases?"
    "presetConfig={%22tags%22:229,%22release%22:122}"
)


@dataclass(frozen=True)
class CapabilityProbe:
    requirement: str
    available: bool
    detail: str


def _module_origin(module) -> str:
    return getattr(module, "__file__", None) or "built-in or namespace package"


def _import_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def probe_legacy_build_capabilities(
    *, include_torchnpugen: bool = True
) -> Tuple[CapabilityProbe, ...]:
    """Import the modules and symbols used by the legacy extension build.

    The imports intentionally happen only when this function is called so the
    default Python-only wheel build remains independent of torch, torch_npu,
    torchnpugen, and the PyTorch C++ extension ABI.
    """

    probes = []

    try:
        cpp_extension = importlib.import_module(CPP_EXTENSION_MODULE)
    except Exception as exc:
        detail = f"module import failed: {_import_error(exc)}"
        probes.extend(
            CapabilityProbe(
                requirement=f"{CPP_EXTENSION_MODULE}.{symbol}",
                available=False,
                detail=detail,
            )
            for symbol in CPP_EXTENSION_SYMBOLS
        )
    else:
        origin = _module_origin(cpp_extension)
        for symbol in CPP_EXTENSION_SYMBOLS:
            if hasattr(cpp_extension, symbol):
                probes.append(
                    CapabilityProbe(
                        requirement=f"{CPP_EXTENSION_MODULE}.{symbol}",
                        available=True,
                        detail=origin,
                    )
                )
            else:
                probes.append(
                    CapabilityProbe(
                        requirement=f"{CPP_EXTENSION_MODULE}.{symbol}",
                        available=False,
                        detail=f"required attribute is missing from {origin}",
                    )
                )

    if include_torchnpugen:
        for module_name in TORCHNPUGEN_MODULES:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                probes.append(
                    CapabilityProbe(
                        requirement=module_name,
                        available=False,
                        detail=_import_error(exc),
                    )
                )
            else:
                probes.append(
                    CapabilityProbe(
                        requirement=module_name,
                        available=True,
                        detail=_module_origin(module),
                    )
                )

    return tuple(probes)


def _version_obj(value: str) -> Optional[Version]:
    try:
        return Version(value.split("+", 1)[0])
    except InvalidVersion:
        return None


def torch_npu_gdn_stream_fix_error(actual: str) -> Optional[str]:
    """Return the legacy GDN stream-policy failure, independently of imports."""

    actual_version = _version_obj(actual)
    if actual_version is None:
        return f"torch_npu has an unsupported version string: {actual}"

    minimum = KNOWN_GDN_STREAM_FIX_MINIMUMS.get(actual_version.base_version)
    if minimum and actual_version >= Version(minimum):
        return None

    if actual_version >= Version(MIN_TORCH_NPU_FUTURE_STREAM_FIX_FAMILY):
        return None

    if minimum is None:
        return None

    requirements = ", ".join(
        f"{family}>={minimum}"
        for family, minimum in KNOWN_GDN_STREAM_FIX_MINIMUMS.items()
    )
    return (
        "torch_npu must come from an Ascend PyTorch release that contains the "
        "GDN aclnn_extension stream fix. Packages from releases before "
        "v26.1.0-beta.1, such as v26.0.0-pytorch2.x, are rejected. "
        f"Expected one of: {requirements}, or "
        f"torch_npu>={MIN_TORCH_NPU_FUTURE_STREAM_FIX_FAMILY} from "
        f"{TORCH_NPU_GDN_STREAM_FIX_RELEASE_URL}; got {actual}."
    )
