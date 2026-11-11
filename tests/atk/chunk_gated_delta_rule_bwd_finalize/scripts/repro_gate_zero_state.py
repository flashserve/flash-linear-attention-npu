"""Issue #921: single token + zero initial state implies dg == 0.
Run on Ascend950 with fla-npu installed: python repro_gate_zero_state.py
"""
import torch
import torch_npu
from fla_npu.ops import ascendc

torch.npu.set_device(0)
rng = torch.Generator().manual_seed(0)
q, k = [torch.randn(1, 1, 2, 128, generator=rng).bfloat16() for _ in range(2)]
v = torch.randn(1, 1, 4, 128, generator=rng).bfloat16()
g = -torch.rand(1, 1, 4, generator=rng)
beta = torch.rand(1, 1, 4, generator=rng)
do = torch.randn(v.shape, generator=rng).bfloat16()
q, k, v, g, beta, do = [x.npu() for x in (q, k, v, g, beta, do)]
options = dict(layout="BSND", chunk_size=64, use_exp2=False,
               use_qk_l2norm_in_kernel=True)

_, _, gcum, a, _, _, qhat, khat, qrstd, krstd = ascendc.chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale=128**-0.5, **options)
_, _, _, _, dg, *_ = ascendc.chunk_gated_delta_rule_bwd(
    qhat, khat, v, gcum, beta, a, do, 128**-0.5,
    q_rstd=qrstd, k_rstd=krstd, **options)
dg = dg.cpu()

# S0=0 => S1=beta1*k1*v1^T: output is independent of g1, so dg1=0.
print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__)
print("dg:", dg.flatten().tolist())
print("max(abs(dg)):", dg.abs().max().item(), "expected: 0")
torch.testing.assert_close(dg, torch.zeros_like(dg), rtol=0, atol=0)
