"""Configurable VDN forward on Ascend 910B."""

import torch

from ._core import forward as _forward
from ._newton import inverse, inverse_sub
from ._short_conv import feature_preprocessor

_FEATURE_BLOCK_BYTES = 128 * 2**20
_CORRECTION_BLOCK_BYTES = 32 * 2**20


def _default_frame_tile(q, delta_rule="vdn_solve"):
    bytes_per_frame = q.size(1) * q.size(2) * q.size(3) * q.element_size()
    tile = 32 if 32 * bytes_per_frame <= _FEATURE_BLOCK_BYTES else 16
    if delta_rule == "newton_schulz":
        # Bound the batch of square FP32 matrices across head counts.
        matrix_bytes_per_frame = q.size(1) * q.size(3) ** 2 * 4
        frames_in_budget = max(1, _CORRECTION_BLOCK_BYTES // matrix_bytes_per_frame)
        tile = min(tile, 1 << (frames_in_budget.bit_length() - 1))
    return tile


def _select_inverse(frame_tile, heads, dim):
    matrix_bytes = frame_tile * heads * dim * dim * 4
    return inverse_sub if matrix_bytes <= _CORRECTION_BLOCK_BYTES else inverse


def vdn_forward(q, k, v, beta, alpha, bounds, text_state=None, *, delta_rule="vdn_solve",
            fuse_silu=False, fuse_qk_l2norm=False, short_conv=None, frame_size=None,
            frame_tile=None, pad_tail=True, value_first=False):
    """Compute the VDN alpha-bridge readout before RMSNorm and output projection.

    q, k and v have shape [frames, heads, tokens, 128]. beta weights each
    token and alpha decays each key channel. bounds holds inclusive window
    endpoints per frame. Both scans start from the optional FP32 value-by-key
    text_state or zeros. The result has the same shape and dtype as q.

    Newton–Schulz assumes normalized keys, beta in [0, 1], and at most
    405 tokens per frame. Disable HF32 for FP32 matrix products. short_conv
    maps feature names to spatial and temporal weights in a frame_size grid.
    """
    if q.device.type != "npu" or q.ndim != 4 or min(q.shape) < 1 or q.size(-1) != 128:
        raise ValueError("optimized forward requires NPU Q/K/V with head_dim=128")
    if frame_tile is None:
        frame_tile = _default_frame_tile(q, delta_rule)
    if type(frame_tile) is not int or frame_tile < 1 or frame_tile & (frame_tile - 1):
        raise ValueError("frame_tile must be a positive power of two")
    inverse_solver = None
    if delta_rule == "newton_schulz":
        if q.size(2) > 405:
            raise ValueError("spectral bound 405 requires at most 405 normalized keys per frame")
        inverse_solver = _select_inverse(frame_tile, q.size(1), q.size(3))
    elif delta_rule != "vdn_solve":
        raise ValueError(f"unsupported delta_rule: {delta_rule!r}")
    # Triton selects its device from the current context, including for JIT compilation.
    with torch.npu.device(q.device):
        preprocess = feature_preprocessor(q, silu=fuse_silu, qk_l2norm=fuse_qk_l2norm,
                                          short_conv=short_conv, frame_size=frame_size)
        return _forward(q, k, v, beta, alpha, bounds, text_state, inverse_solver=inverse_solver,
                        preprocess=preprocess, frame_tile=frame_tile, pad_tail=pad_tail, value_first=value_first)
