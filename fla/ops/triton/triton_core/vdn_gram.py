"""Create the negated FP32 Newton system and its first iterate from the raw Gram tile."""

import torch
import triton
import triton.language as tl


@triton.jit
def _initialize(gram, systems, first, count, diagonal: tl.constexpr, weight: tl.constexpr,
                block: tl.constexpr, programs: tl.constexpr):
    tiles: tl.constexpr = 128 // block
    lane = tl.arange(0, block)
    for job in range(tl.program_id(0), count * tiles * tiles, programs):
        base = (job // (tiles * tiles)).to(tl.int64) * 128 * 128
        row = job // tiles % tiles * block + lane
        col = job % tiles * block + lane
        offset = base + row[:, None] * 128 + col[None, :]
        value = tl.load(gram + offset)
        mirrored = tl.load(gram + base + col[:, None] * 128 + row[None, :])
        eye = (row[:, None] == col[None, :]).to(tl.float32)
        system = (value + tl.trans(mirrored)) * 0.5 + eye
        x = eye * diagonal + system * weight
        tl.store(systems + offset, -system)
        tl.store(first + offset, x)


def initialize(gram, scale):
    system = torch.empty_like(gram)
    first = torch.empty_like(gram)
    block = 64
    cores = triton.runtime.driver.active.utils.get_device_properties(gram.device)["num_vectorcore"]
    programs = min(cores, gram.size(0) * (128 // block) ** 2)
    stream = torch.npu.current_stream(gram.device).npu_stream
    _initialize[(programs,)](gram, system, first, gram.size(0), 2 * scale, -(scale * scale),
                             block, programs, stream=stream, enable_fp_fusion=False)
    return system, first
