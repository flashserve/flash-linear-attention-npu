"""Ascend C stages used by the VDN forward composite on 910B."""

from functools import lru_cache

import torch

from .. import ascendc


@lru_cache(None)
def library(device):
    if device.type != "npu" or "910B" not in torch.npu.get_device_name(device):
        return None
    # A filtered wheel may ship only a subset of the Ascend C operators.
    from fla_npu import _resolve_vendor_dirs
    vendors = _resolve_vendor_dirs()
    for name in ("vdn_scan", "vdn_bridge", "vdn_add_diagonal"):
        config = f"op_impl/ai_core/tbe/kernel/config/ascend910b/{name}.json"
        if not any((vendor / config).is_file() for vendor in vendors):
            return None
    return ascendc


def scan(lib, transition, injection, prefix, suffix):
    prefix[1:].copy_(injection)
    suffix[:-1].copy_(injection)
    lib.npu_vdn_scan(transition, prefix, suffix)


def bridge(lib, prefix, suffix, bounds, logs, rows, dtype):
    state = torch.empty((rows.stop - rows.start, *prefix.shape[1:]), dtype=dtype, device=prefix.device)
    return lib.npu_vdn_bridge(prefix, suffix, bounds, logs, rows.start, state)


def newton_from_first(lib, negative_system, x, eye, iterations):
    two_eye = 2 * eye
    next_x = torch.empty_like(x)
    correction = torch.empty_like(x)
    for _ in range(1, iterations):
        torch.bmm(negative_system, x, out=correction)
        lib.npu_vdn_add_diagonal(correction, two_eye)
        torch.bmm(x, correction, out=next_x)
        x, next_x = next_x, x
    return x
