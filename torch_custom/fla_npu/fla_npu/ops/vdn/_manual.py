"""Manually scheduled four-stage VDN kernel on Ascend 910B3."""
from functools import lru_cache
from pathlib import Path
import os

import torch

from ._newton import ITERATIONS
from ..ascendc import _stable


_PACKAGE = Path(__file__).resolve().parent
_LIBRARY = _PACKAGE / "libvdn_pipeline_device.so"
def _available():
    version = Path(os.getenv("ASCEND_HOME_PATH", "")) / "compiler/version.info"
    return (_LIBRARY.is_file() and version.is_file()
            and "Version=9.2.1" in version.read_text().splitlines())


_HAS_KERNEL = _available()
_GEOMETRIES = {(85, 7), (87, 7), (87, 14)}


@lru_cache(maxsize=None)
def _tiling(device, frames, heads):
    tables = []
    for name in ("c", "a", "b"):
        path = _PACKAGE / "manual_tiling" / f"{name}-{frames}-{heads}.bin"
        data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
        tables.append(data.to(device))
    return tables


def _matches(tensor, device, dtype, shape):
    return (tensor.device == device and tensor.dtype == dtype and tensor.shape == shape
            and tensor.is_contiguous() and tensor.data_ptr() % 32 == 0
            and not tensor.is_neg() and not tensor.is_conj())


def try_forward(q, k, v, beta, alpha, bounds, text_state, *, delta_rule, fuse_silu,
                fuse_qk_l2norm, short_conv, frame_size, frame_tile, pad_tail, value_first):
    frames, heads, tokens, dim = q.shape
    if (not _HAS_KERNEL or (frames, heads) not in _GEOMETRIES or (tokens, dim) != (405, 128)
            or delta_rule != "newton_schulz" or not (fuse_silu and fuse_qk_l2norm and pad_tail)
            or frame_tile not in ((32, 64) if heads == 7 else (32,))
            or not torch.are_deterministic_algorithms_enabled() or torch.npu.matmul.allow_hf32
            or torch.npu.conv.allow_hf32 or torch.npu.is_current_stream_capturing()
            or "910B3" not in torch.npu.get_device_name(q.device)):
        return None
    device, bf = q.device, torch.bfloat16
    if not all(x.ndim == 4 and _matches(x.transpose(1, 2), device, bf, (frames,tokens,heads,dim))
               for x in (q, k, v)):
        return None
    if (not _matches(beta, device, bf, (frames, heads, tokens))
            or not _matches(alpha, device, torch.float32, (frames, heads, dim))
            or bounds.shape != (frames, 2) or bounds.device.type != "cpu" or bounds.dtype != torch.int64
            or (text_state is not None and not _matches(text_state, device, torch.float32, (heads, dim, dim)))):
        return None

    weights = short_conv or {}
    if weights:
        if set(weights) != {"k", "v"} or tuple(frame_size or ()) != (15, 27):
            return None
        for pair in weights.values():
            if (len(pair) != 2 or not _matches(pair[0], device, bf, (heads*dim, 1, 5, 5))
                    or not _matches(pair[1], device, bf, (heads*dim, 5))):
                return None
    tensors = [q,k,v,beta,alpha,*(w for pair in weights.values() for w in pair)]
    if text_state is not None:
        tensors.append(text_state)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return None
    # The device code directly consumes base-format, physically contiguous storage.
    import torch_npu
    if any(torch_npu.get_npu_format(t) not in (0, 2) for t in tensors):
        return None
    return _launch(q,k,v,beta,alpha,bounds,text_state,weights,frame_tile)


def _launch(q, k, v, beta, alpha, bounds, text_state, weights, tile):
    f, h, s, d = q.shape
    row = torch.arange(f, device="cpu")
    lo, hi = bounds.unbind(1)
    if not ((lo >= 0) & (lo <= row) & (row <= hi) & (hi < f)).all():
        raise ValueError("bounds must satisfy 0 <= lo <= frame <= hi < F")
    def bf(*shape):
        return torch.empty(shape, device=q.device, dtype=torch.bfloat16)
    def fp(*shape):
        return torch.empty(shape, device=q.device, dtype=torch.float32)
    def matrix():
        return fp(f,h,d,d)
    inputs = [q,k,v,beta,alpha,bounds.contiguous().to(q.device),text_state if text_state is not None else q,
              *weights.get("k",(q,q)),*weights.get("v",(q,q)),*_tiling(q.device,f,h)]
    if weights:
        spatial_shape = (f*h*s*d,)  # flat GM storage for the packed 5HD panel
        workspace = [bf(*spatial_shape),bf(h*8,25,16,16),bf(*spatial_shape),bf(f,h,s,d),bf(5,h*d),bf(5,h*d)]
    else:
        workspace = [bf(0)] * 6
    # Buffer ordering is shared with the checked C++ binding in params.hpp.
    workspace.extend([
        bf(f,h,s,d),bf(f,h,s,d),bf(f,h,s,d),fp(f,h,s,d),fp(f,h,s,d),
        matrix(),bf(f,h,d,d),matrix(),matrix(),matrix(),matrix(),
        fp(d,d),fp(d,d),fp(20*6*d*d),matrix(),matrix(),
        fp(f+1,h,d,d),fp(f+1,h,d,d),fp(3,f,h,d),fp(f+1,h,d),bf(f,h,d,d),bf(f,h,s,d),
    ])
    panel = 20 if h == 7 else 44
    _stable._op("npu_vdn_pipeline")(*inputs, *workspace, f, h, panel, ITERATIONS - 1,
                                   bool(weights), text_state is not None, _stable._current_stream_ptr())
    return workspace[-1]
