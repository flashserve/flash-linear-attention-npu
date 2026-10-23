"""Regression for the zero-state, single-token GDN gate gradient (issue #921)."""
import argparse

import torch

from chunk_gated_delta_rule_bwd_finalize_cpu import chunk_gated_delta_rule_bwd_finalize_golden


def make_inputs(seed, gate_dtype):
    rng = torch.Generator().manual_seed(seed)
    q, k = [torch.randn(1, 2, 1, 128, generator=rng).bfloat16() for _ in range(2)]
    q, k = [(x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
            for x in (q, k)]
    v = torch.randn(1, 4, 1, 128, generator=rng).bfloat16()
    beta = torch.rand(1, 4, 1, generator=rng).to(gate_dtype)
    g = -torch.rand(1, 4, 1, generator=rng).to(gate_dtype)
    qh, kh = [x.repeat_interleave(2, dim=1).float() for x in (q, k)]
    v_new = (v.float() * beta.float()[..., None]).bfloat16()
    score = (qh * kh).sum(-1, keepdim=True) * 128**-0.5
    do = (2 * score * v_new.float()).bfloat16()
    du = (do.float() * score).bfloat16()
    state = torch.zeros(1, 1, 4, 128, 128, dtype=torch.bfloat16)
    a = torch.zeros(1, 4, 1, 64, dtype=torch.bfloat16)
    a[..., 0] = 1
    return q, k, v, v_new, do, du, g, beta, state, state.clone(), a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'npu'), default='cpu')
    parser.add_argument('--seeds', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == 'npu':
        import torch_npu  # noqa: F401
        from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd_finalize as run
        torch.npu.set_device(0)
    else:
        def run(*xs, **kwargs):
            return chunk_gated_delta_rule_bwd_finalize_golden(*xs, nt_first=True, **kwargs)
    for gate_dtype in (torch.float32, torch.bfloat16):
        for seed in range(args.seeds):
            xs = [x.to(args.device) for x in make_inputs(seed, gate_dtype)]
            actual = run(*xs, use_exp2=False)
            repeat = run(*xs, use_exp2=False)
            for left, right in zip(actual, repeat):
                assert torch.isfinite(left).all(), (seed, gate_dtype)
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            # With zero state, the only gate multiplies zero and cannot affect any output.
            torch.testing.assert_close(actual[-1], torch.zeros_like(actual[-1]), rtol=0, atol=0)
    print(f'PASS: {2 * args.seeds} zero-gate cases, finite and deterministic ({args.device})')


if __name__ == '__main__':
    main()
