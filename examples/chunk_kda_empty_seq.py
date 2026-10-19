"""ChunkKDA 零长度序列看护用例。

`cu_seqlens` 允许相邻项相等，表示零长度序列。零长度序列不贡献 chunk，但仍会
占用 fwd_h 调度器的任务槽；若调度器把“空序列槽”当成流水结束，本核后续的真实
任务会被整体丢弃，表现为对应 (序列, head) 的 `v_new`/`attn_out` 缺失或为 0。

用例做法：同一份输入分别按“含零长度序列”和“去掉零长度序列”两种 `cu_seqlens`
调用 `chunk_kda_fwd`，非空序列的输出必须逐字节一致，且所有输出有限。
零长度序列分别覆盖首段、段间、尾段三种位置。

退出码：0 表示全部一致且有限；1 表示出现差异或非有限值。
"""

import argparse
import sys

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 -- register the accelerator backend
from fla_npu.ops.ascendc import chunk_kda_fwd

NON_EMPTY_LENS = (19, 83, 147)
HEADS = 32
CHUNK_SIZE = 64


def resolve_device(raw):
    text = str(raw)
    if text.isdigit():
        return torch.device("npu:%d" % int(text))
    return torch.device(text)


def forward(tensors, cu_seqlens, device):
    outputs = chunk_kda_fwd(
        tensors["q"], tensors["k"], tensors["v"], tensors["g"], tensors["beta"],
        scale=128 ** -0.5, chunk_size=CHUNK_SIZE, layout="TND",
        initial_state=None, output_final_state=False,
        cu_seqlens=list(cu_seqlens),
        safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
        A_log=tensors["a_log"], dt_bias=tensors["dt_bias"],
        disable_recompute=True, return_intermediate_states=False,
        state_v_first=False,
    )
    if device.type == "npu":
        torch.npu.synchronize()
    return outputs


def make_inputs(args, device):
    tokens = sum(NON_EMPTY_LENS)
    shape = (tokens, HEADS, 128)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    q, k = [
        F.normalize(torch.randn(shape, generator=gen), dim=-1).to(
            device=device, dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    return {
        "q": q,
        "k": k,
        "v": torch.randn(shape, generator=gen).mul_(0.5).to(
            device=device, dtype=torch.bfloat16
        ),
        "g": torch.randn(shape, generator=gen).to(
            device=device, dtype=torch.bfloat16
        ),
        "beta": torch.randn(shape[:-1], generator=gen).sigmoid().to(
            device=device, dtype=torch.bfloat16
        ),
        "a_log": torch.linspace(0, 2.46875, HEADS, dtype=torch.float32).to(device),
        "dt_bias": torch.linspace(
            -8, -2, HEADS * 128, dtype=torch.float32
        ).to(device),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--case-name", default="chunk_kda_empty_seq")
    parser.add_argument("--seed", type=int, default=42)
    args, _unknown = parser.parse_known_args(argv)

    device = resolve_device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)

    boundaries = [0]
    for length in NON_EMPTY_LENS:
        boundaries.append(boundaries[-1] + length)
    variants = {
        "head": [0] + boundaries,
        "middle": boundaries[:2] + [boundaries[1]] + boundaries[2:],
        "tail": boundaries + [boundaries[-1]],
    }

    tensors = make_inputs(args, device)
    baseline = forward(tensors, boundaries, device)
    baseline_map = {
        "o": baseline[0],
        "u": baseline[6],
        "v_new": baseline[9],
    }

    failed = []
    for name, cu in variants.items():
        outputs = forward(tensors, cu, device)
        current = {"o": outputs[0], "u": outputs[6], "v_new": outputs[9]}
        for key, expected in baseline_map.items():
            bad = int((~torch.isfinite(current[key].float())).sum().item())
            if bad:
                print("[%s] %s nonfinite=%d" % (name, key, bad), flush=True)
                failed.append("%s/%s/nonfinite" % (name, key))
                continue
            same = bool(torch.equal(current[key], expected))
            if not same:
                diff = (current[key].float() - expected.float()).abs()
                print(
                    "[%s] %s mismatch count=%d max_abs=%.6g"
                    % (name, key, int((diff > 0).sum().item()), float(diff.max().item())),
                    flush=True,
                )
                failed.append("%s/%s" % (name, key))
        print("[%s] cu_seqlens=%s checked" % (name, cu), flush=True)

    if failed:
        print("empty seq check failed: %s" % ", ".join(failed), file=sys.stderr)
        return 1
    print("empty seq check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
