"""Compare the public KDA forward/backward chain with the ATK dual references."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from atk.configs.case_config import CaseConfig
from atk.tasks.post_process.cv_fused_double_benchmark_compare import CvFusedDoubleBenchmarkAccuracyCompare

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "tests/atk/smoke/cases"
sys.path[:0] = [str(ROOT), str(SMOKE / "chunk_kda_fwd"), str(SMOKE / "chunk_kda_bwd")]
from examples import chunk_kda as model
from executor_chunk_kda_fwd_v2 import PublicForwardApi
from executor_chunk_kda_fwd import _PreparedInputs
import executor_chunk_kda_bwd_A2 as backward

THRESHOLDS = dict(max_re_ratio=5, avg_re_ratio=1.5, root_mean_squared_ratio=1.5)
GRADIENTS = ("dq", "dk", "dv", "dbeta", "dg")


def compare(actual, control, truth, name):
    if not actual.shape == control.shape == truth.shape:
        raise ValueError(f"{name}: shapes differ: {actual.shape}, {control.shape}, {truth.shape}")
    if not all(torch.isfinite(t).all().item() for t in (actual, control, truth)):
        return {"passed": False, "error": "Non-finite output or reference"}
    config = CaseConfig(name=name, inputs=[], standard={"acc": {"cv_fused_double_benchmark": THRESHOLDS}})
    comparator = CvFusedDoubleBenchmarkAccuracyCompare(config)
    comparator.remote_manager = SimpleNamespace(remote_node=SimpleNamespace(
        bm_file=False, name="cpu_control", backend="cpu"))
    comparator.benchmark_standard.init_by_dtype(actual.dtype)
    comparator.benchmark_standard.update(**THRESHOLDS)
    # Public forward references expose the FP64 calculation in the FP32 output ABI.
    if name in ("o", "final_state") and truth.dtype == torch.float64:
        truth = truth.float()
    result = comparator._compute_accuracy(actual.contiguous(), control.contiguous(), truth.contiguous(), name + ".pt")
    if type(result.result) is not bool or not result.new_benchmark_indicate:
        raise ValueError(f"{name}: ATK did not return a verdict and metrics")
    metrics = {k: str(v) if isinstance(v, float) and not math.isfinite(v) else v
               for k, v in result.new_benchmark_indicate.items()}
    return {"passed": result.result, "atk": metrics, "error_info": result.error_info}


def make_inputs(case):
    args = model.parse_args([])
    args.batch, args.tokens, args.query_heads = case["B"], case["T"], case["H"]
    args.value_heads, args.key_dim, args.value_dim = args.query_heads, 128, 128
    args.seed = case["tensor_seed"]
    inputs = model.make_inputs(args, torch.bfloat16, torch.device("cpu"))
    rng = torch.Generator().manual_seed(args.seed + 1)
    grad = torch.randn(inputs["v"].shape, generator=rng).bfloat16()
    return inputs, grad


def npu_chain(inputs, grad, case, cu, device):
    q, k, v, g, beta = [inputs[n].transpose(1, 2).contiguous().to(device)
                        for n in ("q", "k", "v", "g", "beta")]
    q_hat, k_hat = torch.empty_like(q), torch.empty_like(k)
    q_rstd, k_rstd = [torch.empty(q.shape[:-1], dtype=torch.float32, device=device) for _ in range(2)]
    bias = inputs["dt_bias"].to(device)
    options = dict(cu_seqlens=cu, safe_gate=True, lower_bound=case["lower_bound"], use_gate_in_kernel=True,
                   A_log=inputs["A_log"].to(device), use_exp2=True, disable_recompute=True,
                   use_beta_sigmoid_in_kernel=False)
    output, state, gk, aqk, akk, w, _, qg, kg, v_new, h, _ = model.chunk_kda_fwd(
        q, k, v, g, beta, 128 ** -0.5, 64, layout="BNSD", dt_bias=bias, output_final_state=True,
        use_qk_l2norm_in_kernel=True, q_hat_out=q_hat, k_hat_out=k_hat,
        q_rstd_out=q_rstd, k_rstd_out=k_rstd, **options)
    grads = model.chunk_kda_bwd(
        q_hat, k_hat, inputs["v"].to(device), inputs["beta"].to(device), gk, aqk, akk, w, qg, kg,
        v_new, h, grad.to(device), 128 ** -0.5, q_rstd=q_rstd, k_rstd=k_rstd, layout="BSND",
        raw_g=inputs["g"].to(device), chunk_size=64, dt_bias=bias.reshape(case["H"], 128), **options)
    torch.npu.synchronize()
    caches = dict(q=q_hat, k=k_hat, q_rstd=q_rstd, k_rstd=k_rstd, gk=gk, Aqk=aqk, Akk=akk,
                  w=w, qg=qg, kg=kg, v_new=v_new, h=h)
    return [t.cpu() for t in (output, state, *grads[:5])], {n: t.cpu() for n, t in caches.items()}


def run_case(case, cu, device, result):
    inputs, grad = make_inputs(case)
    actual, caches = npu_chain(inputs, grad, case, cu, device)
    values = {n: inputs[n].transpose(1, 2).contiguous() for n in ("q", "k", "v", "g", "beta")}
    spec = dict(B=case["B"], H=case["H"], HV=case["H"], T=case["T"], K=128, V=128, layout="BNSD",
                chunk_size=64, scale=128 ** -0.5, epsilon=1e-6, safe_gate=True, lower_bound=case["lower_bound"],
                use_qk_l2norm_in_kernel=True, use_exp2=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=False,
                allow_neg_eigval=False, state_v_first=False, output_final_state=True, disable_recompute=True,
                return_intermediate_states=False, q_dtype="bf16", g_dtype="fp32", beta_dtype="bf16")
    prepared = _PreparedInputs(**values, A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
                               initial_state=None, cu_seqlens=cu, chunk_indices=None)
    control = PublicForwardApi._run_cpu_control(prepared, spec)[:2]
    truth = PublicForwardApi._run_fp64(prepared, spec)[:2]
    metrics = result["metrics"] = {}
    for i, name in enumerate(("o", "final_state")):
        metrics[name] = compare(actual[i], control[i], truth[i], name)
    del control, truth

    caches.update(v=values["v"], beta=values["beta"], raw_g=values["g"], d_o=grad.transpose(1, 2).contiguous(),
                  A_log=inputs["A_log"], dt_bias=inputs["dt_bias"].reshape(case["H"], 128))
    spec.update(mode="varlen" if cu else "dense", layout="BSND", with_l2=True, with_bias=True,
                beta_mode="plain", raw_g_dtype="fp32", seqlens=[b-a for a, b in zip(cu, cu[1:])] if cu else None)
    control = backward._cast_cpu_outputs_to_abi(backward.cpu_reference(caches, spec, high_precision=False), spec)
    truth = backward.cpu_reference(caches, spec, high_precision=True)
    for i, name in enumerate(GRADIENTS):
        a, low, high = actual[i + 2], control[i], truth[i]
        if cu and a.ndim == high.ndim + 1 and a.shape[0] == 1:
            low, high = low.unsqueeze(0), high.unsqueeze(0)
        metrics[name] = compare(a, low, high, name)
    return all(item["passed"] for item in metrics.values())
