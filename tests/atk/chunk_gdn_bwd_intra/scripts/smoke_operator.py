"""在 NPU 上直接调用完整算子，排查 ABI 与同步问题。"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]

# GVA 比例上限按平台声明，与算子侧 aclnn 参数校验同口径。
GVA_LIMIT = {2201: 16, 3510: 4}


def rejection_matches(message: str, hv: int, hk: int, arch: int) -> bool:
    # 断言 aclnn 参数校验层（OP_LOGE errno 直报）的报文：实际 HV/HK、
    # 该平台的合法整数比范围及其判定来源。tiling 侧报文格式不同，
    # 负向 smoke 以 workspace-query 阶段稳定透出为准。
    limit = GVA_LIMIT[arch]
    patterns = (
        r"aclnnChunkGdnBwdIntraGetWorkspaceSize failed:\s*161002\b",
        rf"\bHV\s*=\s*{hv}\b",
        rf"\bHK\s*=\s*{hk}\b",
        r"incorrect value",
        rf"integer ratio in\s*\[\s*1\s*,\s*{limit}\s*\]",
        r"current SoC arch",
    )
    return all(re.search(pattern, message) is not None for pattern in patterns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--g-dtype", default="fp32", choices=("bf16", "fp32"))
    parser.add_argument("--beta-dtype", default="fp32", choices=("bf16", "fp32"))
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hk", type=int)
    parser.add_argument("--hv", type=int)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--use-exp2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--expect-reject", action="store_true")
    parser.add_argument("--expected-arch", type=int, choices=(2201, 3510))
    args = parser.parse_args(argv)

    if args.group < 1 or args.batch < 1 or args.tokens < 1:
        parser.error("--group, --batch and --tokens must be positive")
    args.hk = args.hk if args.hk is not None else (
        4 if args.group in {1, 2} else 2 if args.group == 3 else 1
    )
    args.hv = args.hv if args.hv is not None else args.hk * args.group
    if args.hk < 1 or args.hv < 1:
        parser.error("--hk and --hv must be positive")
    if args.expect_reject:
        if args.expected_arch is None:
            parser.error("--expect-reject requires --expected-arch")
        limit = GVA_LIMIT[args.expected_arch]
        if args.hv % args.hk == 0 and args.hv // args.hk <= limit:
            parser.error("--expect-reject requires an invalid HV/HK ratio for --expected-arch")
    elif not 1 <= args.group <= GVA_LIMIT[2201] or args.hv != args.hk * args.group:
        parser.error("positive mode requires --group in [1, 16] and --hv == --hk * --group")
    return args


def main() -> None:
    args = parse_args()
    if args.expect_reject:
        os.environ["FLA_NPU_STABLE_LAUNCH"] = "inline"
    import torch

    sys.path.insert(0, str(OP_DIR))
    from executor_chunk_gdn_bwd_intra import _cpu_ref, build_inputs

    hk, hv = args.hk, args.hv
    spec = {
        "dtype": args.dtype,
        "g_dtype": args.g_dtype,
        "beta_dtype": args.beta_dtype,
        "B": args.batch,
        "HK": hk,
        "HV": hv,
        "T": args.tokens,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": args.use_exp2,
        "seed": args.seed,
    }
    inputs = build_inputs(spec, torch.device("npu"))

    from fla_npu.ops import ascendc

    if args.expect_reject:
        try:
            ascendc.chunk_gdn_bwd_intra(
                inputs["q"], inputs["k"], inputs["v"], inputs["g"],
                inputs["beta"], inputs["A"], inputs["d_o"],
                inputs["scale"], inputs["chunk_size"],
                use_exp2=inputs["use_exp2"],
            )
        except Exception as exc:  # noqa: BLE001 - 错误形态随版本可能是 RuntimeError/ValueError
            message = str(exc)
            ok = rejection_matches(message, hv, hk, args.expected_arch)
            print("rejected_as_expected:" if ok else "ERROR: unexpected rejection:", flush=True)
            print({"hv": hv, "hk": hk, "arch": args.expected_arch, "message": message}, flush=True)
            raise SystemExit(0 if ok else 3)
        print("ERROR: expected rejection but operator returned normally", flush=True)
        raise SystemExit(2)

    watchdog = subprocess.Popen(
        ["sh", "-c", 'sleep 60; kill -KILL "$1"', "operator-watchdog", str(os.getpid())]
    )
    try:
        print("operator_call_begin", flush=True)
        outputs = ascendc.chunk_gdn_bwd_intra(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A"],
            inputs["d_o"],
            inputs["scale"],
            inputs["chunk_size"],
            use_exp2=inputs["use_exp2"],
        )
        print("operator_call_returned", flush=True)
        torch.npu.synchronize()
        print("operator_synchronize_done", flush=True)
    finally:
        watchdog.terminate()
        watchdog.wait()

    stats = [
        {
            "shape": tuple(output.shape),
            "finite": bool(torch.isfinite(output.float()).all().item()),
            "abs_max": float(output.float().abs().max().item()),
        }
        for output in outputs
    ]
    print(stats)
    if not args.compare:
        return

    cpu_inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
    refs = _cpu_ref(cpu_inputs)
    for index, (output, ref) in enumerate(zip(outputs, refs)):
        actual = output.float().cpu()
        expected = ref.float()
        error = (actual - expected).abs()
        denom = expected.abs().clamp_min(1e-8)
        flat_index = int(error.argmax().item())
        max_index = []
        for size in reversed(error.shape):
            max_index.append(flat_index % size)
            flat_index //= size
        max_index = tuple(reversed(max_index))
        print(
            {
                "output": index,
                "max_abs": float(error.max().item()),
                "max_index": max_index,
                "actual_at_max": float(actual[max_index].item()),
                "expected_at_max": float(expected[max_index].item()),
                "mean_abs": float(error.mean().item()),
                "max_rel": float((error / denom).max().item()),
                "allclose_5e-3": bool(
                    torch.allclose(actual, expected, rtol=5e-3, atol=5e-3)
                ),
                "head_max_abs": error.amax(dim=(0, 2, 3)).tolist(),
                "chunk_max_abs": [
                    float(error[..., start : start + 64, :].max().item())
                    for start in range(0, args.tokens, 64)
                ],
            }
        )


if __name__ == "__main__":
    main()
