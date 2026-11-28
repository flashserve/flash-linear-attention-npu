"""Optional spatial short conv and Vector-fused temporal conv for VDN."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .vdn_layout import activate_head_major, activate_kv_head_major, token_major
from .vdn_features import ROW_TILE, VECTOR_PROGRAMS, _activate_values, _source_round, activate_tile


@triton.jit
def _tap_round(x, kind: tl.constexpr):
    if kind == tl.bfloat16:
        # Keep each round-to-nearest conversion visible to the compiler.
        # Rebuild FP32 from the resulting BF16 bits before the next operation.
        hi = x.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        return (hi << 16).to(tl.float32, bitcast=True)
    return x


@triton.jit
def _temporal_activate(src, weight, dst, beta, WEIGHT_VALUE: tl.constexpr,
                       HEADS, TOKENS, D: tl.constexpr,
                       FIRST_OUT, HALO_FRAMES,
                       OUT_FRAMES, SILU: tl.constexpr, NORM: tl.constexpr,
                       BLOCK: tl.constexpr, PROGRAMS: tl.constexpr):
    token_lane = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    kind: tl.constexpr = src.dtype.element_ty
    tiles_per_head = tl.cdiv(TOKENS, BLOCK)
    for task in range(tl.program_id(0), OUT_FRAMES * HEADS * tiles_per_head, PROGRAMS):
        frame = task // (HEADS * tiles_per_head)
        head = task // tiles_per_head % HEADS
        token = task % tiles_per_head * BLOCK + token_lane
        feature = head * D + d
        accum = tl.full((BLOCK, D), 0.0, tl.float32)
        for tap in tl.static_range(5):
            f = frame + FIRST_OUT + tap - 2
            addr = ((f * HEADS + head) * TOKENS + token[:, None]) * D + d[None, :]
            x = tl.load(src + addr, (token[:, None] < TOKENS) & (f >= 0) & (f < HALO_FRAMES),
                        other=0).to(tl.float32)
            w = tl.load(weight + tap * HEADS * D + feature).to(tl.float32)
            product = _tap_round(x * w[None, :], kind)
            accum = _tap_round(accum + product, kind)
        y = _activate_values(accum, kind, SILU, NORM, BLOCK)
        if WEIGHT_VALUE:
            offset = (frame * HEADS + head) * TOKENS + token
            coeff = tl.load(beta + offset, token < TOKENS, other=0).to(tl.float32)
            y = _source_round(y, kind) * coeff[:, None]
        dst_addr = ((frame * HEADS + head) * TOKENS + token[:, None]) * D + d[None, :]
        tl.store(dst + dst_addr, y, token[:, None] < TOKENS)


def _short_tile(tensor, rows, frame_size, weights, *, silu, l2norm, beta=None):
    frames, heads, tokens, dim = tensor.shape
    channels = heads * dim
    start, end = max(rows.start - 2, 0), min(rows.stop + 2, frames)
    gh, gw = frame_size
    halo = tensor[start:end].transpose(2, 3).reshape(end - start, channels, gh, gw)
    spatial = F.conv2d(halo, weights[0], padding=2, groups=channels)
    del halo
    spatial = spatial.reshape(end - start, heads, dim, tokens).transpose(2, 3).contiguous()
    out = torch.empty((rows.stop - rows.start, heads, tokens, dim),
                      device=tensor.device, dtype=tensor.dtype)
    out_frames = rows.stop - rows.start
    programs = min(out_frames * heads * triton.cdiv(tokens, ROW_TILE), VECTOR_PROGRAMS)
    stream = torch.npu.current_stream(tensor.device).npu_stream
    beta_ptr = out if beta is None else beta
    _temporal_activate[(programs,)](
        spatial, weights[1], out, beta_ptr, beta is not None, heads, tokens, dim,
        rows.start - start, end - start, out_frames, silu, l2norm, ROW_TILE, programs, stream=stream)
    return out


def feature_preprocessor(q, *, silu, qk_l2norm, short_conv, frame_size):
    """short_conv maps q/k/v to (spatial_weight, temporal_weight)."""
    if not silu and not qk_l2norm and not short_conv:
        return None
    weights = {}
    if short_conv:
        if (frame_size is None or len(frame_size) != 2 or min(frame_size) < 1
                or frame_size[0] * frame_size[1] != q.size(2)):
            raise ValueError("short_conv needs frame_size=(height,width) matching S")
        channels = q.size(1) * q.size(3)
        for key, (sp, tm) in short_conv.items():
            if key not in ("q", "k", "v") or sp.shape != (channels, 1, 5, 5):
                raise ValueError("short_conv expects q/k/v keys with spatial weights [H*d,1,5,5]")
            if tm.shape not in ((channels, 5), (channels, 1, 5)):
                raise ValueError("temporal weights must be [H*d,5] or [H*d,1,5]")
            if any(w.device != q.device or w.dtype != q.dtype for w in (sp, tm)):
                raise ValueError("short_conv weights must share Q/K/V's device and dtype")
            weights[key] = (sp, tm.reshape(channels, 5).transpose(0, 1).contiguous())

    return _FeaturePreprocessor(silu, qk_l2norm, weights, frame_size)


class _FeaturePreprocessor:
    def __init__(self, silu, qk_l2norm, weights, frame_size):
        self.silu = silu
        self.qk_l2norm = qk_l2norm
        self.weights = weights
        self.frame_size = frame_size

    def __call__(self, tensor, rows, projection):
        l2norm = self.qk_l2norm and projection in ("q", "k")
        if projection in self.weights:
            return _short_tile(tensor, rows, self.frame_size, self.weights[projection],
                               silu=self.silu, l2norm=l2norm)
        if token_major(tensor):
            return activate_head_major(tensor[rows], silu=self.silu, l2norm=l2norm, block=64)
        return activate_tile(tensor[rows], silu=self.silu, l2norm=l2norm)

    def key_value(self, k, v, beta, rows):
        if "v" in self.weights and beta.dtype == v.dtype and beta.is_contiguous():
            key = self(k, rows, "k")
            value = _short_tile(v, rows, self.frame_size, self.weights["v"],
                                silu=self.silu, l2norm=False, beta=beta)
            return key, value, True
        if (self.silu and self.qk_l2norm and "k" not in self.weights
                and "v" not in self.weights and token_major(k) and token_major(v)):
            key, weighted_value = activate_kv_head_major(k[rows], v[rows], beta)
            return key, weighted_value, True
        return self(k, rows, "k"), self(v, rows, "v"), False
