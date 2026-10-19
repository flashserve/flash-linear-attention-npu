"""ChunkKDA varLen 尾块 workspace 看护用例。

构造 4 条都带尾块的 TND 序列（219/260/272/198 token，chunk=64，共补 203 行），
HV=96，直接调用 `fla_npu.ops.ascendc.chunk_kda_fwd` 并检查全部输出有限。

该形态会放大 post-Wu 暂存区的越界写入：越界元素数 = 补行数 * HV * K，
按 token 口径分配时约 10 MB，足以覆盖后面的 fwd_h workspace 与 output scratch，
表现为 o/u/v_new/h 出现 1e38/NaN。用例只做有限性判定，不做精度判定。

退出码：0 表示所有输出有限；1 表示出现非有限值。
"""

import argparse
import sys

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 -- register the accelerator backend
from fla_npu.ops.ascendc import chunk_kda_fwd

TAIL_BOUNDARIES = (0, 219, 479, 751, 949)
HEADS = 96
CHUNK_SIZE = 64


def build_chunk_indices(cu_seqlens, chunk_size):
    """按 fla 约定生成 chunk_indices: (NT, 2)，每行 [seq_idx, chunk_idx_in_seq]。"""
    cu = torch.as_tensor(cu_seqlens, dtype=torch.long)
    lens = cu[1:] - cu[:-1]
    per_seq = (lens + chunk_size - 1) // chunk_size
    flat = torch.cat([torch.arange(n, dtype=torch.long) for n in per_seq.tolist()])
    seq_idx = flat.eq(0).cumsum(0) - 1
    return torch.stack([seq_idx, flat], dim=1)


def resolve_device(raw):
    text = str(raw)
    if text.isdigit():
        return torch.device("npu:%d" % int(text))
    return torch.device(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--case-name", default="chunk_kda_tail_scratch")
    parser.add_argument("--seed", type=int, default=42)
    args, _unknown = parser.parse_known_args(argv)

    device = resolve_device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)

    boundaries = list(TAIL_BOUNDARIES)
    tokens = boundaries[-1]
    shape = (tokens, HEADS, 128)
    chunk_indices = tuple(
        tuple(row) for row in build_chunk_indices(boundaries, CHUNK_SIZE).tolist()
    )

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    q, k = [
        F.normalize(torch.randn(shape, generator=gen), dim=-1).to(
            device=device, dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    v = torch.randn(shape, generator=gen).mul_(0.5).to(
        device=device, dtype=torch.bfloat16
    )
    g = torch.randn(shape, generator=gen).to(device=device, dtype=torch.bfloat16)
    beta = torch.randn(shape[:-1], generator=gen).sigmoid().to(device=device)
    a_log = torch.linspace(0, 2.46875, HEADS, dtype=torch.float32).to(device=device)
    dt_bias = torch.linspace(
        -8, -2, HEADS * 128, dtype=torch.float32
    ).to(device=device)

    outputs = chunk_kda_fwd(
        q, k, v, g, beta,
        scale=128 ** -0.5, chunk_size=CHUNK_SIZE, layout="TND",
        initial_state=None, output_final_state=False,
        cu_seqlens=boundaries, chunk_indices=chunk_indices,
        safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
        A_log=a_log, dt_bias=dt_bias, disable_recompute=True,
        return_intermediate_states=False, state_v_first=False,
    )
    if device.type == "npu":
        torch.npu.synchronize()

    names = ("o", "final_state", "gk", "Aqk", "Akk", "w", "u", "qg", "kg",
             "v_new", "h", "initial_state")
    failed = []
    for index, tensor in enumerate(outputs):
        if tensor is None:
            continue
        bad = int((~torch.isfinite(tensor.float())).sum().item())
        label = names[index] if index < len(names) else "out%d" % index
        print("%-12s shape=%s nonfinite=%d" % (label, tuple(tensor.shape), bad),
              flush=True)
        if bad:
            failed.append(label)
    if failed:
        print("tail scratch check failed: %s" % ", ".join(failed), file=sys.stderr)
        return 1
    print("tail scratch check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
