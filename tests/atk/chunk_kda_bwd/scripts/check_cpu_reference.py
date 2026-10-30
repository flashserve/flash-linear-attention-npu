import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_chunk_kda_bwd import (
    _head_major_output, _token_major_input, build_inputs, run_cpu,
)


def one_step_reference(values, spec):
    names = ["q", "k", "v", "beta", "raw_g", "A_log"]
    if "dt_bias" in values:
        names.append("dt_bias")
    leaves = {name: values[name].detach().double().requires_grad_(True) for name in names}
    q, k = leaves["q"], leaves["k"]
    if spec.get("norm"):
        q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.0e-6)).to(torch.bfloat16).double()
        k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)).to(torch.bfloat16).double()
    raw = leaves["raw_g"]
    if "dt_bias" in leaves:
        raw = raw + leaves["dt_bias"].view(1, -1, 1, 128)
    gate = -5.0 * torch.sigmoid(leaves["A_log"].exp().view(1, -1, 1, 1) * raw)
    beta = leaves["beta"]
    state = k[:, :, 0].unsqueeze(-1) * (beta[:, :, 0].unsqueeze(-1) * leaves["v"][:, :, 0]).unsqueeze(-2)
    state = state + gate[:, :, 0].exp().unsqueeze(-1) * 0.0
    output = torch.einsum("bhkv,bhk->bhv", state, q[:, :, 0]) / math.sqrt(128)
    loss = (output * values["d_o"].double()[:, :, 0]).sum()
    gradients = torch.autograd.grad(loss, [leaves[name] for name in names])
    if spec.get("cu_seqlens"):
        gradients = tuple(g.squeeze(0) if i < 5 else g for i, g in enumerate(gradients))
    return tuple(g.float() for g in gradients)


def check_layout_roundtrip():
    vector = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4)
    scalar = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
    for packed in (False, True):
        for value, is_scalar in ((vector, False), (scalar, True)):
            original = value.clone()
            token_major = _token_major_input(value, packed, scalar=is_scalar)
            expected_token_shape = (
                (3, 2) if is_scalar else (3, 2, 4)
            ) if packed else (
                (1, 3, 2) if is_scalar else (1, 3, 2, 4)
            )
            assert token_major.shape == expected_token_shape
            restored = _head_major_output(token_major, packed, scalar=is_scalar)
            expected = value.squeeze(0) if packed else value
            assert restored.shape == expected.shape
            assert torch.equal(restored, expected)
            assert torch.equal(value, original)
    print("CPU dense/packed layout roundtrip OK")


def main():
    generator = torch.Generator().manual_seed(20261027)
    shape = (1, 2, 1, 128)
    values = {
        name: torch.randn(shape, generator=generator, dtype=torch.float64) * scale
        for name, scale in (("q", 0.05), ("k", 0.05), ("v", 0.05),
                            ("raw_g", 1.25), ("d_o", 51.2))
    }
    values["beta"] = torch.rand(shape[:-1], generator=generator, dtype=torch.float64) * 0.8 + 0.1
    values["A_log"] = torch.randn(2, generator=generator, dtype=torch.float64) * 0.12
    values["dt_bias"] = torch.randn(2, 128, generator=generator, dtype=torch.float64) * 1.65 - 3.0
    cases = (
        (False, False, True, "fp32"),
        (True, False, True, "fp32"),
        (False, True, False, "fp32"),
        (False, False, True, "bf16"),
    )
    for norm, packed, with_bias, raw_g_dtype in cases:
        current = dict(values)
        if raw_g_dtype == "bf16":
            current["raw_g"] = current["raw_g"].to(torch.bfloat16).double()
        if not with_bias:
            current.pop("dt_bias")
        spec = {
            "norm": norm, "cu_seqlens": [0, 1] if packed else None,
            "raw_g_dtype": raw_g_dtype,
        }
        actual = run_cpu(current, spec)
        expected = one_step_reference(current, spec)
        assert len(actual) == len(expected) == (7 if with_bias else 6)
        for index, (a, e) in enumerate(zip(actual, expected)):
            assert a.shape == e.shape, (norm, packed, index, a.shape, e.shape)
            assert a.dtype == torch.float32
            torch.testing.assert_close(a, e, rtol=1.0e-5, atol=1.0e-5)
        print(
            f"CPU one-step OK norm={norm} packed={packed} "
            f"raw_g={raw_g_dtype} outputs={len(actual)}"
        )
    spec = dict(B=1, H=8, T=64, seed=0, dtype="bf16", beta_dtype="fp32", dt_bias=True)
    first = build_inputs(spec, torch.device("cpu"), high_precision=True)
    second = build_inputs(spec, torch.device("cpu"), high_precision=True)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert first["q"].shape == (1, 8, 64, 128)
    assert first["beta"].shape == (1, 8, 64)
    print("CPU deterministic input construction OK")

    bf16_spec = dict(spec, raw_g_dtype="bf16")
    bf16_golden = build_inputs(bf16_spec, torch.device("cpu"), high_precision=True)
    assert torch.equal(bf16_golden["raw_g"], first["raw_g"].to(torch.bfloat16).double())
    runtime = build_inputs(bf16_spec, torch.device("cpu"), high_precision=False)
    assert runtime["raw_g"].dtype == torch.bfloat16
    provided = build_inputs(
        bf16_spec, torch.device("cpu"), high_precision=True, provided=runtime,
    )
    assert provided["raw_g"].dtype == torch.float64
    assert torch.equal(provided["raw_g"], runtime["raw_g"].double())
    print("CPU BF16 raw_g quantization and provided-input consumption OK")
    check_layout_roundtrip()


if __name__ == "__main__":
    main()
