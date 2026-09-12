"""ChunkKdaFwdPrepare 稳定 ctypes/aclnn 通路的最小运行辅助。"""

from __future__ import annotations

import os
from typing import Any


OUTPUT_NAMES = (
    "gk",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "qg_scaled",
    "q_hat",
    "k_hat",
    "q_rstd",
    "k_rstd",
    "beta_eff",
)


def require_npu_test_environment():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        return None, None

    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("FLA_NPU_RUN_OPERATOR_TESTS=1 but no NPU is available")
    device_id = int(os.environ.get("FLA_NPU_DEVICE", "0"))
    torch.npu.set_device(device_id)
    return torch, torch.device(f"npu:{device_id}")


def _dtype(torch, name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _layout_shapes(case: dict[str, Any]):
    shape = case["shape"]
    batch = int(shape["B"])
    key_heads = int(shape["HK"])
    value_heads = int(shape["HV"])
    tokens = int(shape["T"])
    key_dim = int(shape["K"])
    value_dim = int(shape["V"])
    layout = case["layout"]

    if layout == "BNSD":
        return (
            (batch, key_heads, tokens, key_dim),
            (batch, value_heads, tokens, value_dim),
            (batch, value_heads, tokens, key_dim),
            (batch, value_heads, tokens),
        )
    if layout == "BSND":
        return (
            (batch, tokens, key_heads, key_dim),
            (batch, tokens, value_heads, value_dim),
            (batch, tokens, value_heads, key_dim),
            (batch, tokens, value_heads),
        )
    if layout == "NTD":
        return (
            (key_heads, tokens, key_dim),
            (value_heads, tokens, value_dim),
            (value_heads, tokens, key_dim),
            (value_heads, tokens),
        )
    if layout == "TND":
        return (
            (tokens, key_heads, key_dim),
            (tokens, value_heads, value_dim),
            (tokens, value_heads, key_dim),
            (tokens, value_heads),
        )
    raise ValueError(f"unsupported layout: {layout}")


def _randn(torch, shape, dtype, device, generator, scale=0.05):
    value = torch.randn(shape, dtype=torch.float32, generator=generator)
    return value.mul(scale).to(dtype=dtype, device=device).contiguous()


def build_inputs(torch, device, case: dict[str, Any]):
    q_shape, v_shape, g_shape, beta_shape = _layout_shapes(case)
    dtype = case["dtype"]
    attrs = case["attrs"]
    optional = case["optional_inputs"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(case["seed"]))

    q = _randn(torch, q_shape, torch.bfloat16, device, generator)
    k = _randn(torch, q_shape, torch.bfloat16, device, generator)
    v = _randn(torch, v_shape, torch.bfloat16, device, generator)
    if attrs["use_gate_in_kernel"]:
        g = _randn(torch, g_shape, _dtype(torch, dtype["g"]), device, generator)
    else:
        g = -torch.rand(g_shape, dtype=torch.float32, generator=generator)
        g = g.mul_(0.018).add_(0.002)
        g = g.to(dtype=_dtype(torch, dtype["g"]), device=device).contiguous()
    beta = _randn(
        torch,
        beta_shape,
        _dtype(torch, dtype["beta"]),
        device,
        generator,
        scale=0.5,
    )

    value_heads = int(case["shape"]["HV"])
    a_log = None
    if optional["a_log"] is not None:
        a_log = torch.linspace(
            -6.0, -2.0, value_heads, dtype=torch.float32, device=device
        )
    dt_bias = None
    if optional["dt_bias"] is not None:
        dt_bias = _randn(
            torch,
            (value_heads * int(case["shape"]["K"]),),
            torch.float32,
            device,
            generator,
            scale=0.25,
        )
    return q, k, v, g, beta, a_log, dt_bias


def expected_output_specs(torch, case: dict[str, Any]):
    shape = case["shape"]
    batch = int(shape["B"])
    key_heads = int(shape["HK"])
    value_heads = int(shape["HV"])
    tokens = int(shape["T"])
    key_dim = int(shape["K"])
    value_dim = int(shape["V"])
    chunk_size = int(case["attrs"]["chunk_size"])
    rank4 = case["layout"] in {"BNSD", "BSND"}
    prefix = (batch,) if rank4 else ()
    gate_shape = prefix + (value_heads, tokens, key_dim)
    key_value_shape = prefix + (value_heads, tokens, key_dim)
    value_shape = prefix + (value_heads, tokens, value_dim)
    key_shape = prefix + (key_heads, tokens, key_dim)
    value_scalar_shape = prefix + (value_heads, tokens)
    key_scalar_shape = prefix + (key_heads, tokens)
    matrix_shape = prefix + (value_heads, tokens, chunk_size)
    return (
        (gate_shape, torch.float32),
        (matrix_shape, torch.bfloat16),
        (matrix_shape, torch.bfloat16),
        (key_value_shape, torch.bfloat16),
        (value_shape, torch.bfloat16),
        (key_value_shape, torch.bfloat16),
        (key_value_shape, torch.bfloat16),
        (key_value_shape, torch.bfloat16),
        (key_shape, torch.bfloat16),
        (key_shape, torch.bfloat16),
        (key_scalar_shape, torch.float32),
        (key_scalar_shape, torch.float32),
        (value_scalar_shape, torch.float32),
    )


def run_stable_aclnn_case(torch, device, case: dict[str, Any]):
    from fla_npu.ops.ascendc import chunk_kda_fwd_prepare

    q, k, v, g, beta, a_log, dt_bias = build_inputs(torch, device, case)
    attrs = case["attrs"]
    optional = case["optional_inputs"]
    outputs = chunk_kda_fwd_prepare(
        q,
        k,
        v,
        g,
        beta,
        float(attrs["scale"]),
        layout=str(attrs["layout"]),
        chunk_size=int(attrs["chunk_size"]),
        epsilon=float(attrs["epsilon"]),
        use_qk_l2norm_in_kernel=bool(attrs["use_qk_l2norm_in_kernel"]),
        use_gate_in_kernel=bool(attrs["use_gate_in_kernel"]),
        use_beta_sigmoid_in_kernel=bool(
            attrs["use_beta_sigmoid_in_kernel"]
        ),
        allow_neg_eigval=bool(attrs["allow_neg_eigval"]),
        safe_gate=bool(attrs["safe_gate"]),
        lower_bound=float(attrs["lower_bound"]),
        use_exp2=bool(attrs["use_exp2"]),
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=optional["cu_seqlens"],
        chunk_indices=optional["chunk_indices"],
        backward_mode=str(attrs.get("backward_mode", "save")),
    )
    torch.npu.synchronize()
    return outputs


def check_output_contract(torch, case: dict[str, Any], outputs):
    expected = expected_output_specs(torch, case)
    present_outputs = set(
        case.get("expect", {}).get("present_outputs", OUTPUT_NAMES)
    )
    if len(outputs) != len(OUTPUT_NAMES):
        raise AssertionError(
            f"{case['id']}: expected {len(OUTPUT_NAMES)} outputs, got {len(outputs)}"
        )
    for name, output, (shape, dtype) in zip(OUTPUT_NAMES, outputs, expected):
        if name not in present_outputs:
            if output is not None:
                raise AssertionError(
                    f"{case['id']}:{name} must be None for "
                    f"backward_mode={case['attrs'].get('backward_mode', 'save')}"
                )
            continue
        if output is None:
            raise AssertionError(f"{case['id']}:{name} is unexpectedly None")
        if tuple(output.shape) != shape:
            raise AssertionError(
                f"{case['id']}:{name} shape {tuple(output.shape)} != {shape}"
            )
        if output.dtype != dtype:
            raise AssertionError(
                f"{case['id']}:{name} dtype {output.dtype} != {dtype}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise AssertionError(f"{case['id']}:{name} contains NaN or Inf")
