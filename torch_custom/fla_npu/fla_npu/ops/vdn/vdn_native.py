"""Optional Stable-ABI dispatcher for the 910B VDN kernels."""
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def library(device):
    if "910B" not in torch.npu.get_device_name(device):
        return None
    path = Path(__file__).with_name("libfla_vdn_910b.so")
    if not path.is_file():
        return None
    torch.ops.load_library(str(path))
    return torch.ops.fla_vdn_910b


def scan(lib, transition, injection, prefix, suffix):
    prefix[1:].copy_(injection)
    suffix[:-1].copy_(injection)
    with torch.npu.device(transition.device):
        stream = torch.npu.current_stream(transition.device).npu_stream
        lib.scan(transition, prefix, suffix, stream)


def bridge(lib, prefix, suffix, bounds, logs, rows, dtype):
    count = rows.stop - rows.start
    state = torch.empty((count, *prefix.shape[1:]), dtype=dtype, device=prefix.device)
    with torch.npu.device(prefix.device):
        stream = torch.npu.current_stream(prefix.device).npu_stream
        lib.bridge(prefix, suffix, bounds, logs, state, rows.start, stream)
    return state


def newton_from_first(lib, negative_system, x, eye, iterations):
    two_eye = 2 * eye
    next_x = torch.empty_like(x)
    correction = torch.empty_like(x)
    with torch.npu.device(negative_system.device):
        for _ in range(1, iterations):
            torch.bmm(negative_system, x, out=correction)
            # Flush queued BMM submissions before the inline kernel launch.
            stream = torch.npu.current_stream(negative_system.device).npu_stream
            lib.add_diagonal(correction, two_eye, stream)
            torch.bmm(x, correction, out=next_x)
            x, next_x = next_x, x
    return x

