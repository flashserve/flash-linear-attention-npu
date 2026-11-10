"""Capabilities published by the OPP that supplies the recurrent kernels.

The Python wheel and the OPP are separate artifacts: the wheel installs
``fla_npu`` plus a vendor tree, but a user can replace or overlay the vendor
tree without touching Python.  A hard-coded length limit in a caller therefore
describes the wheel it was written against, not the kernel that will run.

The kernel publishes its per-sequence window next to its sources in
``op_kernel/capabilities.json``; the OPP install copies that file into
``<vendor>/op_impl/ai_core/tbe/<impl>/ascendc/<op>/``.  This module reads the
published value from the vendor that actually wins the load order, so callers
never have to guess.  Kernels that predate the file are recognized by their
single-file layout and their ``MAX_MTP`` definition.

``max_verify_len=None`` means the kernel consumes a sequence in windows and
imposes no per-sequence cap.  An *unpublished* capability is different: it
means the installed OPP says nothing, and the caller must probe or stay
conservative.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from functools import lru_cache

_GDN_OP = "recurrent_gated_delta_rule"
_IMPL_ROOT = ("op_impl", "ai_core", "tbe")
_ARCH22_PREFIXES = ("Ascend910B", "Ascend910C", "Ascend910_93")
_ARCH35_PREFIX = "Ascend950"
_LEGACY_MAX_MTP = re.compile(r"\bMAX_MTP\s*=\s*(\d+)\s*;")


@dataclass(frozen=True)
class RecurrentGdnCapability:
    """What the loaded OPP says about ``recurrent_gated_delta_rule``.

    ``max_verify_len`` is the largest number of tokens one sequence may carry
    in a single call.  ``None`` means the kernel has no per-sequence cap.
    ``published`` is False when the OPP does not describe the operator at all;
    in that case ``max_verify_len`` is unknown, not unbounded.
    """

    arch: str
    max_verify_len: int | None
    published: bool
    source: str
    path: str | None = None


def _arch_from_device_name(name: str) -> str:
    if name.startswith(_ARCH35_PREFIX):
        return "arch35"
    if name.startswith(_ARCH22_PREFIXES):
        return "arch22"
    raise LookupError(
        "recurrent_gated_delta_rule has no published capability for device "
        f"{name!r}; expected an Ascend910B/Ascend910C/Ascend910_93 (arch22) "
        "or Ascend950 (arch35) device."
    )


def _device_name(device_index: int | None) -> str:
    import torch

    if device_index is None:
        device_index = torch.npu.current_device()
    return str(torch.npu.get_device_name(device_index))


def _vendor_dirs() -> list[pathlib.Path]:
    # Imported lazily: this module is also loaded by offline tests that have no
    # fla_npu package context, and the package import initializes the OPP.
    import fla_npu

    resolve_many = getattr(fla_npu, "_resolve_vendor_dirs", None)
    if resolve_many is not None:
        return [pathlib.Path(path) for path in resolve_many()]
    # Wheels predating the overlay chain expose only the singular resolver,
    # which already returns the base vendor or the highest-priority overlay.
    return [pathlib.Path(fla_npu._resolve_vendor_dir())]


def _op_dirs(vendor_dir: pathlib.Path) -> list[pathlib.Path]:
    root = pathlib.Path(vendor_dir).joinpath(*_IMPL_ROOT)
    # The implementation directory name is derived from the vendor name and is
    # not part of the public contract, so glob it instead of hard-coding it.
    return sorted(path for path in root.glob(f"*/ascendc/{_GDN_OP}") if path.is_dir())


def _published_capability(op_dir: pathlib.Path, arch: str) -> RecurrentGdnCapability | None:
    path = op_dir / "capabilities.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    arches = data.get("arches")
    if not isinstance(arches, dict):
        raise ValueError(f"{path} must contain an 'arches' object.")
    entry = arches.get(arch)
    if not isinstance(entry, dict):
        raise LookupError(
            f"{path} does not publish a capability for {arch}; the OPP that "
            "wins the load order does not describe this device."
        )
    value = entry.get("max_verify_len")
    if value is None:
        return RecurrentGdnCapability(arch, None, True, "opp-capabilities", str(path))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{path}: {arch}.max_verify_len must be a positive integer or "
            f"null, got {value!r}."
        )
    return RecurrentGdnCapability(arch, value, True, "opp-capabilities", str(path))


def _legacy_capability(op_dir: pathlib.Path, arch: str) -> RecurrentGdnCapability | None:
    """Read the pre-V2 single-file kernel, which shipped its own window cap.

    Only arch22 had a single-file kernel; arch35 always used the tiled V2
    implementation, so a legacy directory cannot describe an Ascend950.
    """

    if arch != "arch22":
        return None
    path = op_dir / f"{_GDN_OP}.h"
    if not path.is_file():
        return None
    match = _LEGACY_MAX_MTP.search(path.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        return None
    return RecurrentGdnCapability(
        arch, int(match.group(1)), True, "legacy-kernel-header", str(path)
    )


def _resolve_capability(arch: str, vendor_dirs: list[pathlib.Path]) -> RecurrentGdnCapability:
    for vendor_dir in vendor_dirs:
        for op_dir in _op_dirs(vendor_dir):
            published = _published_capability(op_dir, arch)
            if published is not None:
                return published
            legacy = _legacy_capability(op_dir, arch)
            if legacy is not None:
                return legacy
            # This vendor wins the load order and ships the operator, so a
            # lower-priority vendor's capability would not describe it.
            return RecurrentGdnCapability(arch, None, False, "unpublished", str(op_dir))
    return RecurrentGdnCapability(arch, None, False, "unpublished", None)


@lru_cache(maxsize=16)
def _cached_capability(device_index: int) -> RecurrentGdnCapability:
    arch = _arch_from_device_name(_device_name(device_index))
    try:
        vendor_dirs = _vendor_dirs()
    except OSError:
        return RecurrentGdnCapability(arch, None, False, "unpublished", None)
    return _resolve_capability(arch, vendor_dirs)


def recurrent_gdn_capability(device_index: int | None = None) -> RecurrentGdnCapability:
    """Capability record for the OPP that will run ``recurrent_gated_delta_rule``.

    ``LookupError`` is raised when the device is not an A2/A3/A5 device.  A
    missing publication is reported through ``published=False`` rather than an
    exception, because callers decide whether to probe or to fail.
    """

    if device_index is None:
        device_index = _current_device_index()
    return _cached_capability(int(device_index))


def _current_device_index() -> int:
    import torch

    return int(torch.npu.current_device())


def recurrent_gdn_max_verify_len(device_index: int | None = None) -> int | None:
    """Largest per-sequence verify window accepted by the loaded kernel.

    ``None`` means the kernel imposes no per-sequence cap.  ``LookupError``
    means the installed OPP does not publish a value (for example an overlay
    vendor that predates ``capabilities.json``); callers should probe once.
    """

    capability = recurrent_gdn_capability(device_index)
    if not capability.published:
        raise LookupError(
            "the installed fla_npu OPP does not publish "
            f"recurrent_gated_delta_rule capabilities for {capability.arch}; "
            "probe the operator once or stay on the conservative path."
        )
    return capability.max_verify_len


def validate_recurrent_gdn_lengths(
    seq_lengths: list[int], device_index: int | None = None
) -> RecurrentGdnCapability | None:
    """Reject a per-sequence window the loaded kernel would silently skip.

    Returns the capability used for the decision, or ``None`` when the device
    is not recognized (the operator itself reports a clearer error then).
    Unpublished capabilities are not checked: without a published window the
    caller's start-up probe owns that decision.
    """

    try:
        capability = recurrent_gdn_capability(device_index)
    except LookupError:
        return None
    if not capability.published or capability.max_verify_len is None:
        return capability
    longest = max(seq_lengths, default=0)
    if longest > capability.max_verify_len:
        raise ValueError(
            "recurrent_gated_delta_rule accepts at most "
            f"{capability.max_verify_len} tokens per sequence on "
            f"{capability.arch}, got {longest}. The installed OPP skips a "
            "longer sequence without writing its output or advancing its "
            f"state ({capability.source}: {capability.path}). Reduce the "
            "speculative window or install an OPP with a wider window."
        )
    return capability
