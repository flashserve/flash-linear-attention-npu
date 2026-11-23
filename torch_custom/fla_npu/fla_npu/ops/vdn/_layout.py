"""Prepare and repack token-major feature tiles on the available Vector cores."""

import torch
import triton
import triton.language as tl

from ._features import _activate_values, _source_round, activate_tile


@triton.jit
def _tile_offsets(task, HEADS, TOKENS, D: tl.constexpr, BLOCK: tl.constexpr):
    tiles = tl.cdiv(TOKENS, BLOCK)
    frame = task // (HEADS * tiles)
    head = task // tiles % HEADS
    token = task % tiles * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, D)
    src = ((frame * TOKENS + token[:, None]) * HEADS + head) * D + col[None, :]
    dst = ((frame * HEADS + head) * TOKENS + token[:, None]) * D + col[None, :]
    return frame, head, token, src, dst


@triton.jit
def _head_major(src, dst, FRAMES, HEADS,
                TOKENS, D: tl.constexpr, SILU: tl.constexpr, NORM: tl.constexpr,
                BLOCK: tl.constexpr, PROGRAMS: tl.constexpr):
    tasks = FRAMES * HEADS * tl.cdiv(TOKENS, BLOCK)
    for task in range(tl.program_id(0), tasks, PROGRAMS):
        frame, head, token, src_addr, dst_addr = _tile_offsets(task, HEADS, TOKENS, D, BLOCK)
        mask = token[:, None] < TOKENS
        x = tl.load(src + src_addr, mask, other=0).to(tl.float32)
        y = _activate_values(x, src.dtype.element_ty, SILU, NORM, BLOCK)
        tl.store(dst + dst_addr, y, mask)


@triton.jit
def _head_major_kv(k, v, beta, key, value, FRAMES, HEADS, TOKENS, D: tl.constexpr,
                   BLOCK: tl.constexpr, PROGRAMS: tl.constexpr,
                   BF: tl.constexpr, BH: tl.constexpr, BS: tl.constexpr):
    tasks = FRAMES * HEADS * tl.cdiv(TOKENS, BLOCK)
    for task in range(tl.program_id(0), tasks, PROGRAMS):
        frame, head, token, src_addr, dst_addr = _tile_offsets(task, HEADS, TOKENS, D, BLOCK)
        mask = token[:, None] < TOKENS
        x = tl.load(k + src_addr, mask, other=0).to(tl.float32)
        y = _activate_values(x, k.dtype.element_ty, True, True, BLOCK)
        tl.store(key + dst_addr, y, mask)
        x = tl.load(v + src_addr, mask, other=0).to(tl.float32)
        y = _activate_values(x, v.dtype.element_ty, True, False, BLOCK)
        weight = tl.load(beta + frame * BF + head * BH + token * BS, token < TOKENS, other=0)
        weight = _source_round(weight.to(tl.float32), v.dtype.element_ty)
        tl.store(value + dst_addr, y * weight[:, None], mask)


def token_major(src):
    return not src.is_contiguous() and src.transpose(1, 2).is_contiguous()


def activate_kv_head_major(k, v, beta, block=64):
    frames, heads, tokens, dim = k.shape
    key = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    value = torch.empty_like(key)
    stream = torch.npu.current_stream(k.device).npu_stream
    cores = triton.runtime.driver.active.utils.get_device_properties(k.device)["num_vectorcore"]
    programs = min(frames * heads * triton.cdiv(tokens, block), cores)
    _head_major_kv[(programs,)](k, v, beta, key, value, frames, heads, tokens, dim,
                                block, programs, *beta.stride(), stream=stream)
    return key, value


def activate_head_major(src, *, silu, l2norm, block=32):
    if (not silu and not l2norm) or not token_major(src):
        return activate_tile(src, silu=silu, l2norm=l2norm)
    frames, heads, tokens, dim = src.shape
    result = torch.empty(src.shape, device=src.device, dtype=src.dtype)
    cores = triton.runtime.driver.active.utils.get_device_properties(src.device)["num_vectorcore"]
    programs = min(frames * heads * triton.cdiv(tokens, block), cores)
    stream = torch.npu.current_stream(src.device).npu_stream
    _head_major[(programs,)](src, result, frames, heads, tokens, dim, silu, l2norm,
                             block, programs, stream=stream)
    return result
