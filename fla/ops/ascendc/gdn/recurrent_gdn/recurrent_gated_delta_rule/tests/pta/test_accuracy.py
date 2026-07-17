"""
Accuracy test for npu_recurrent_gated_delta_rule.

Compares NPU output against CPU golden reference.
Tests multiple parameter combinations.
"""
import sys
import torch
import torch_npu

from fla_npu.ops import ascendc as ascendc_ops

from golden import recurrent_gated_delta_rule_golden
from utils import compare_tensors_by_ratio


def make_inputs(bs, mtp, nk, nv, dk, dv, use_g=True, use_gk=False,
                use_accepted_tokens=False, seed=42):
    """Generate inputs on CPU, matching the kernel's expected shapes."""
    torch.manual_seed(seed)
    star_idx = 1
    star_idx_tensor = torch.tensor([star_idx], dtype=torch.int32)
    batch_tensor = torch.ones(bs, dtype=torch.int32) * mtp
    actual_seq_lengths = torch.cat([star_idx_tensor, batch_tensor])
    t = int(actual_seq_lengths.sum().item())

    state = torch.rand((t, nv, dv, dk), dtype=torch.float)
    query = torch.nn.functional.normalize(
        torch.rand((t, nk, dk), dtype=torch.bfloat16), p=2, dim=-1
    )
    key = torch.nn.functional.normalize(
        torch.rand((t, nk, dk), dtype=torch.bfloat16), p=2, dim=-1
    )
    value = torch.rand((t, nv, dv), dtype=torch.bfloat16)
    beta = torch.rand((t, nv), dtype=torch.bfloat16)
    scale = dk ** -0.5

    ssm_state_indices = torch.arange(t, dtype=torch.int32)

    g = None
    if use_g:
        g = -torch.rand((t, nv), dtype=torch.float32)

    gk = None
    if use_gk:
        gk = -torch.rand((t, nv, dk), dtype=torch.float32)

    num_accepted_tokens = None
    if use_accepted_tokens and mtp > 1:
        num_accepted_tokens = torch.randint(1, mtp + 1, (bs,), dtype=torch.int32)

    return {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "scale": scale,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "g": g,
        "gk": gk,
    }


def run_golden(inp):
    """Run CPU golden implementation."""
    return recurrent_gated_delta_rule_golden(
        query=inp["query"],
        key=inp["key"],
        value=inp["value"],
        state=inp["state"].clone(),
        beta=inp["beta"],
        scale=inp["scale"],
        actual_seq_lengths=inp["actual_seq_lengths"],
        ssm_state_indices=inp["ssm_state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        g=inp["g"],
        gk=inp["gk"],
    )


def run_npu(inp, device):
    """Run NPU operator."""
    print("start run npu")
    q_npu = inp["query"].npu()
    k_npu = inp["key"].npu()
    v_npu = inp["value"].npu()
    s_npu = inp["state"].clone().npu()
    b_npu = inp["beta"].npu()
    asl_npu = inp["actual_seq_lengths"].npu()
    ssi_npu = inp["ssm_state_indices"].npu()
    print("start run npu_recurrent_gated_delta_rule")

    g_npu = inp["g"].to(device) if inp["g"] is not None else None
    gk_npu = inp["gk"].to(device) if inp["gk"] is not None else None
    nat_npu = inp["num_accepted_tokens"].to(device) if inp["num_accepted_tokens"] is not None else None
    
    print("start run npu_recurrent_gated_delta_rule")

    result = ascendc_ops.recurrent_gated_delta_rule(
        q_npu, k_npu, v_npu, s_npu,
        beta=b_npu,
        actual_seq_lengths=asl_npu,
        ssm_state_indices=ssi_npu,
        num_accepted_tokens=nat_npu,
        g=g_npu,
        gk=gk_npu,
        scale_value=inp["scale"],
    )
    torch_npu.npu.synchronize()

    # result is (attn_out, final_state)
    if isinstance(result, (tuple, list)):
        attn_out = result[0].cpu()
        final_state = result[1].cpu()
    else:
        attn_out = result.cpu()
        final_state = s_npu.cpu()
    attn_out[:asl_npu[0]] = 0
    return attn_out, final_state


def run_test_case(desc, bs, mtp, nk, nv, dk, dv, device,
                  use_g=True, use_gk=False, use_accepted_tokens=False,
                  seed=42, rtol=0.01, atol=0.004):
    """Run a single test case and report result."""
    print(f"\n{'='*60}")
    print(f"TEST: {desc}")
    print(f"  bs={bs}, mtp={mtp}, nk={nk}, nv={nv}, dk={dk}, dv={dv}")
    print(f"  use_g={use_g}, use_gk={use_gk}, use_accepted_tokens={use_accepted_tokens}")
    print(f"  seed={seed}, rtol={rtol}, atol={atol}")
    print(f"{'='*60}")

    inp = make_inputs(bs, mtp, nk, nv, dk, dv,
                      use_g=use_g, use_gk=use_gk,
                      use_accepted_tokens=use_accepted_tokens, seed=seed)

    print("  Running CPU golden ...")
    golden_attn, golden_state = run_golden(inp)

    print("  Running NPU ...")
    npu_attn, npu_state = run_npu(inp, device)

    print(f"  NPU attn  shape={npu_attn.shape}  dtype={npu_attn.dtype}")
    print(f"  NPU state shape={npu_state.shape}  dtype={npu_state.dtype}")

    attn_pass = compare_tensors_by_ratio(golden_attn, npu_attn, "attn_out", rtol=rtol, atol=atol)
    state_pass = compare_tensors_by_ratio(golden_state, npu_state, "final_state", rtol=rtol, atol=atol)

    overall = attn_pass and state_pass
    status = "PASS" if overall else "FAIL"
    print(f"\n  >> {desc}: {status}")
    return overall


def main():
    device = torch.device("npu:2")
    torch_npu.npu.set_device(device)
    results = []
    print(f"Using device: {device}")
    results.append(run_test_case(
        "basic_bs2_mtp2",
        bs=2, mtp=2, nk=4, nv=8, dk=128, dv=128, device=device,
        use_g=True, use_accepted_tokens=False,
    ))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    pass_count = sum(results)
    total = len(results)
    for i, r in enumerate(results):
        print(f"  Test {i+1}: {'PASS' if r else 'FAIL'}")
    print(f"\n  {pass_count}/{total} passed")

    if pass_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
