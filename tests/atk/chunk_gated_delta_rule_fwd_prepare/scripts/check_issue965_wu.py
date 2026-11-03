"""Stage-isolation diagnostic for #965 (not a replacement for ATK acceptance).

The default shape has 245 tasks / 62 packs. On a 28-core launch the singleton
last pack leaves AIV1 idle while it must still drain its preceding pack.
Recompute W/U from the *same* device A/K/g to exclude inverse/L2norm rounding.
"""
import argparse
import math

import torch


def check(args, seed, output_a):
    from fla_npu.ops import ascendc

    torch.manual_seed(seed)
    b, hk, hv, t = args.batch, args.heads, args.heads * args.head_ratio, args.tokens
    q, k = [(torch.rand(b, hk, t, 128) * .002 - .001).bfloat16() for _ in range(2)]
    v = (torch.rand(b, hv, t, 128) * .002 - .001).bfloat16()
    g, beta = [torch.rand(b, hv, t) * .002 - .001 for _ in range(2)]
    inputs = [x.to(f'npu:{args.device}') for x in (q, k, v, g, beta)]
    # A is collected in an identical companion invocation when output_a=False.
    kw = dict(chunk_size=64, use_qk_l2norm_in_kernel=True, use_exp2=args.exp2)
    collected = ascendc.npu_chunk_gated_delta_rule_fwd_prepare(*inputs, output_a=True, **kw)
    result = collected if output_a else ascendc.npu_chunk_gated_delta_rule_fwd_prepare(
        *inputs, output_a=False, **kw)
    torch.npu.synchronize()
    kn, gn, w, u = [result[i].cpu().float() for i in (1, 5, 6, 7)]
    a = collected[8].cpu().float()
    for start in range(0, t, 64):
        end = min(start + 64, t)
        gc = gn[:, :, start:end]
        eg = (gc * math.log(2)).exp() if args.exp2 else gc.exp()
        bc = beta[:, :, start:end]
        kc = kn[:, :, start:end].repeat_interleave(args.head_ratio, dim=1)
        vb = (v[:, :, start:end].float() * bc[..., None]).bfloat16().float()
        kb = (kc * (bc * eg)[..., None]).bfloat16().float()
        ac = a[:, :, start:end, :end-start]
        for name, actual, expected in (
            ('w', w[:, :, start:end], (ac @ kb).bfloat16().float()),
            ('u', u[:, :, start:end], (ac @ vb).bfloat16().float()),
        ):
            # Compare per tile so a corrupt head cannot hide in a global norm.
            error = (actual - expected).abs().amax(dim=(-1, -2))
            limit = expected.abs().amax(dim=(-1, -2)) * .02 + 1e-12
            if not bool((error <= limit).all()):
                bad = torch.nonzero(error > limit).tolist()
                raise AssertionError(f'{name}: seed={seed}, output_a={output_a}, '
                                     f'tokens=[{start},{end}), bad [batch,head]={bad}, '
                                     f'max_abs={error.max().item():.8g}')
    print(f'PASS seed={seed} output_a={output_a} B={b} HK={hk} HV={hv} T={t} exp2={args.exp2}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--heads', type=int, default=7)
    parser.add_argument('--head-ratio', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--tokens', type=int, default=448)
    parser.add_argument('--seeds', type=int, nargs='+', default=[8253, 17, 29])
    parser.add_argument('--exp2', action='store_true')
    args = parser.parse_args()
    import torch_npu  # noqa: F401
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    for seed in args.seeds:
        for output_a in (False, True):
            check(args, seed, output_a)


if __name__ == '__main__':
    main()
