#!/usr/bin/env python3
"""Regress #1057 through the A2/A3 public Stable-ABI backward entry.

Run with an installed A2/A3 wheel and matching OPP. Positive cases are execution
smokes, not numerical accuracy checks. Negative cases must fail at ACLNN phase 1.
"""
import argparse

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import _stable


def inputs(layout, disable_recompute):
    torch.manual_seed(1057)
    heads, tokens, width, chunk = 4, 64, 128, 64
    packed = layout in ("NTD", "TND")
    token_major = layout in ("BSND", "TND")
    prefix = () if packed else (1,)
    head_shape = prefix + (heads, tokens, width)
    token_shape = prefix + (tokens, heads, width)
    external = token_shape if token_major else head_shape

    def tensor(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device="npu") * 0.01

    case = dict(
        q=tensor(external), k=tensor(external), v=tensor(external),
        beta=tensor(prefix + (tokens, heads)),
        gk=tensor(head_shape, torch.float32),
        Aqk=tensor(prefix + (heads, tokens, chunk)),
        Akk=tensor(prefix + (heads, tokens, chunk)),
        w=tensor(head_shape), qg=tensor(head_shape), kg=tensor(head_shape),
        v_new=tensor(head_shape), h=tensor(prefix + (1, heads, width, width)),
        d_o=tensor(token_shape), raw_g=tensor(token_shape),
        A_log=tensor((heads,), torch.float32), scale=width ** -0.5,
        chunk_size=chunk, lower_bound=-5.0, use_gate_in_kernel=True,
        disable_recompute=disable_recompute, use_exp2=True, layout=layout,
    )
    if packed:
        case.update(cu_seqlens=[0, tokens], chunk_indices=[0, 0])
    if not disable_recompute:
        for name in ("gk", "w", "qg", "kg", "v_new", "h"):
            case[name] = None
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    from fla_npu.ops.ascendc._kda_policy import _is_kda_bwd_arch22
    if not _is_kda_bwd_arch22(torch.empty(1, device="npu")):
        raise RuntimeError("This regression requires the A2/A3 backward path.")

    for layout in ("BNSD", "BSND", "NTD", "TND"):
        saved = inputs(layout, True)
        out = _stable.npu_chunk_kda_bwd(**saved, safe_gate=True)
        torch.npu.synchronize()
        for result, name in zip(out[:5], ("q", "k", "v", "beta", "raw_g")):
            assert result.shape == saved[name].shape, (layout, name, result.shape)
            assert result.dtype == saved[name].dtype, (layout, name, result.dtype)
            assert torch.isfinite(result).all().item(), (layout, name)
        print(f"PASS {layout} safe_gate=True saved-cache execution", flush=True)

        for disable_recompute in (True, False):
            case = saved if disable_recompute else inputs(layout, False)
            try:
                _stable.npu_chunk_kda_bwd(**case, safe_gate=False)
                torch.npu.synchronize()
            except RuntimeError as error:
                message = str(error)
                assert "aclnnChunkKdaBwdGetWorkspaceSize failed: 161002" in message, message
                assert "safe_gate" in message and "false" in message and "true on A2/A3" in message, message
                assert "EZ0024" in message or "EZ0002" in message, message
                assert "561103" not in message and "Failed to execute tiling" not in message, message
            else:
                raise AssertionError("safe_gate=False was accepted")
            print(f"PASS {layout} disable_recompute={disable_recompute} safe_gate=False -> 161002",
                  flush=True)
    print("PASS 4 execution smokes and 8 invalid-parameter regressions", flush=True)


if __name__ == "__main__":
    main()
