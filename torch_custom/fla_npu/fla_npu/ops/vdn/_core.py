"""VDN-Solve forward on preprocessed tensors [F, H, S, d]."""

import torch

from . import vdn_native

from ._cholesky import solve
from ._newton import ITERATIONS, inverse_sub

# On 910B2C the diagonal update amortizes its launches at eight waves of 48 diagonal tiles.
_NATIVE_NEWTON_MIN_BATCH = 192


def frame_blocks(frames, limit):
    if type(limit) is not int or limit < 1 or limit & (limit - 1):
        raise ValueError("frame_tile must be a positive power of two")
    start = 0
    while start < frames:
        count = min(limit, 1 << ((frames - start).bit_length() - 1))
        yield slice(start, start + count)
        start += count


def _frame_plan(frames, limit, pad_tail):
    if not pad_tail:
        return tuple((r, r.stop - r.start) for r in frame_blocks(frames, limit))
    if type(limit) is not int or limit < 1 or limit & (limit - 1):
        raise ValueError("frame_tile must be a positive power of two")
    result = []
    for start in range(0, frames, limit):
        stop = min(start + limit, frames)
        width = 1 << ((stop - start - 1).bit_length())
        result.append((slice(start, stop), width))
    return tuple(result)


def _pad_frames(x, width):
    if x.size(0) == width:
        return x
    padded = x.new_zeros((width, *x.shape[1:]))
    padded[:x.size(0)].copy_(x)
    return padded


def _value_gram(v, k, beta, value_weighted):
    weighted_value = v if value_weighted else v * beta.to(v.dtype)
    return torch.bmm(weighted_value.mT, k)


def _scan_update(bias, state, matrix, *, out):
    return torch.add(torch.bmm(state, matrix), bias, out=out)


def _factor(k, v, beta, eye, *, inverse_solve=None, value_weighted=False, injection_out=None, value_first=False):
    k = k.flatten(0, 1).contiguous()
    v = v.flatten(0, 1)
    beta = beta.flatten(0, 1).unsqueeze(-1)
    pending_b = _value_gram(v, k, beta, value_weighted) if value_first else None
    k32 = k.float()
    a = torch.bmm((k32 * beta.float()).mT, k32)
    native_inverse = None
    if (inverse_solve is inverse_sub and a.device.type == "npu" and a.size(-1) == 128
            and a.size(0) >= _NATIVE_NEWTON_MIN_BATCH and not torch.npu.matmul.allow_hf32):
        native_inverse = vdn_native.library(a.device)
    if native_inverse is None:
        a = (a + a.mT) * 0.5
    if pending_b is None:
        pending_b = _value_gram(v, k, beta, value_weighted)
    b = pending_b.float()
    del pending_b
    if native_inverse is not None:
        from ._gram import initialize
        system, first = initialize(a)
    else:
        system = a + eye
    del a, k32, k, v, beta
    if inverse_solve is not None:
        if native_inverse is None:
            inverse = inverse_solve(system, eye)
        else:
            inverse = vdn_native.newton_from_first(native_inverse, system, first, eye, ITERATIONS)
            del first
        del system
        return inverse, torch.bmm(b, inverse, out=injection_out)
    lower_inv = solve(system)
    del system
    inverse = torch.bmm(lower_inv.mT, lower_inv)
    return inverse, torch.bmm(b, inverse, out=injection_out)


@torch.no_grad()
def forward(q, k, v, beta, alpha, bounds, text_state=None, *, inverse_solver, preprocess,
            frame_tile, pad_tail, value_first):
    """Return the pre-RMSNorm readout [F,H,S,d] in the Q/K/V dtype.

    Inputs Q/K/V can be strided FP32 or BF16. beta is [F,H,S] and alpha is
    FP32 [F,H,d]. bounds is int64 [F,2] with inclusive, post-anchor indices,
    satisfying 0 <= lo <= frame <= hi < F. The optional FP32 text_state is
    [H,d,d] in value-key order, already scaled by 0.5.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape or min(q.shape) < 1:
        raise ValueError("Q, K and V must have one nonempty [F,H,S,d] shape")
    frames, heads, tokens, dim = q.shape
    if q.dtype not in (torch.float32, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Q, K and V must share FP32 or BF16 dtype")
    if any(x.device != q.device for x in (k, v, beta, alpha)):
        raise ValueError("Q, K, V, beta and alpha must share one device")
    if beta.shape != (frames, heads, tokens) or beta.dtype not in (q.dtype, torch.float32):
        raise ValueError("beta must be [F,H,S] with the Q dtype or FP32")
    if alpha.shape != (frames, heads, dim) or alpha.dtype != torch.float32:
        raise ValueError("alpha must be FP32 [F,H,d]")
    if bounds.shape != (frames, 2) or bounds.dtype != torch.int64:
        raise ValueError("bounds must be int64 [F,2]")
    if bounds.device.type == "cpu":
        row = torch.arange(frames)
        lo, hi = bounds.unbind(1)
        if not ((0 <= lo) & (lo <= row) & (row <= hi) & (hi < frames)).all():
            raise ValueError("bounds must satisfy 0 <= lo <= frame <= hi < F")
    elif bounds.device != q.device:
        raise ValueError("bounds must reside on CPU or the Q device")
    if text_state is not None and (text_state.shape != (heads, dim, dim)
                                   or text_state.device != q.device
                                   or text_state.dtype != torch.float32):
        raise ValueError("text_state must be FP32 [H,d,d] on the Q device")
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _forward(q, k, v, beta, alpha, bounds, text_state, inverse_solver=inverse_solver, preprocess=preprocess,
                        frame_tile=frame_tile, pad_tail=pad_tail, value_first=value_first)


def _forward(q, k, v, beta, alpha, bounds, text_state, *, inverse_solver, preprocess,
            frame_tile, pad_tail, value_first):
    frames, heads, _, dim = q.shape
    eye = torch.eye(dim, device=q.device, dtype=torch.float32)
    blocks = _frame_plan(frames, frame_tile, pad_tail)
    transition = eye.new_empty((frames, heads, dim, dim))
    injection = torch.empty_like(transition)
    for rows, width in blocks:
        beta_tile = beta[rows]
        if preprocess is None:
            key, value, value_weighted = k[rows], v[rows], False
        else:
            key, value, value_weighted = preprocess.key_value(k, v, beta_tile, rows)
        real = rows.stop - rows.start
        if real < width:
            key, value, beta_tile = (_pad_frames(t, width) for t in (key, value, beta_tile))
        injection_out = injection[rows].flatten(0, 1) if real == width else None
        inv, inj = _factor(key, value, beta_tile, eye,
                           inverse_solve=inverse_solver, value_weighted=value_weighted, injection_out=injection_out,
                           value_first=value_first)
        del key, value
        scale = alpha[rows].unsqueeze(-1)
        inverse = inv[:real * heads].unflatten(0, (-1, heads))
        torch.mul(scale, inverse, out=transition[rows])
        if injection_out is None:
            injection[rows].copy_(inj[:real * heads].unflatten(0, (-1, heads)))
        del inverse, injection_out
    del inv, inj

    # The extra row on each side represents the virtual text state at the boundary.
    prefix = eye.new_empty((frames + 1, heads, dim, dim))
    suffix = torch.empty_like(prefix)
    if text_state is None:
        prefix[0].zero_()
        suffix[frames].zero_()
    else:
        prefix[0].copy_(text_state)
        suffix[frames].copy_(text_state)
    native = vdn_native.library(q.device)
    if native is not None and not torch.npu.matmul.allow_hf32:
        vdn_native.scan(native, transition, injection, prefix, suffix)
    else:
        for frame in range(frames):
            _scan_update(injection[frame], prefix[frame], transition[frame], out=prefix[frame + 1])
        for frame in range(frames - 1, -1, -1):
            _scan_update(injection[frame], suffix[frame + 1], transition[frame], out=suffix[frame])
    del transition, injection

    use_bridge = native is not None and bounds.device.type == "cpu"
    bounds = bounds.to(q.device)
    logs = alpha.clamp_min(1e-12).log()
    log_prefix = torch.cat((torch.zeros_like(logs[:1]), logs.cumsum(0)), dim=0)
    if not use_bridge:
        lo, hi = bounds.unbind(1)
        after = hi + 1
        row = torch.arange(frames, device=q.device)
        left = (log_prefix.index_select(0, row + 1) - log_prefix.index_select(0, lo)).exp()
        right = (log_prefix.index_select(0, after) - log_prefix.index_select(0, row)).exp()
    output = q.new_empty(q.shape)
    for rows, width in blocks:
        if use_bridge:
            state = vdn_native.bridge(native, prefix, suffix, bounds, log_prefix, rows, q.dtype)
        else:
            state = (prefix.index_select(0, lo[rows]) * left[rows].unsqueeze(-2)
                     + suffix.index_select(0, after[rows]) * right[rows].unsqueeze(-2))
        query = q[rows] if preprocess is None else preprocess(q, rows, "q")
        real = rows.stop - rows.start
        state = state.to(q.dtype)
        if real < width:
            query, state = (_pad_frames(t, width) for t in (query, state))
        writeback = output[rows].flatten(0, 1) if real == width else None
        out = torch.bmm(query.flatten(0, 1).contiguous(),
                        state.flatten(0, 1).mT, out=writeback)
        if writeback is None:
            output[rows].copy_(out[:real * heads].unflatten(0, (-1, heads)))
        del query
    return output
