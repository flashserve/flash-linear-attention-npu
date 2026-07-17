"""JSON-driven NPU accuracy and contract runner for bwd_dhu."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu
from test_bwd_dhu import chunk_gated_delta_rule_bwd_dhu_cpu


def _manifest():
    configured = os.environ.get("FLA_NPU_CASE_MANIFEST")
    if configured:
        path = Path(configured)
    else:
        path = Path(__file__).resolve().parents[3] / "tests" / "op_cases" / "chunk_gated_delta_rule_bwd_dhu.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_cases(manifest):
    selected = {item for item in os.environ.get("FLA_NPU_CASE_IDS", "").split(",") if item}
    return [case for case in manifest["cases"] if not selected or case["id"] in selected]


def _dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _present(optional, name):
    return optional.get(name) not in (None, False)


def _inputs(case):
    shape = case["shape"]
    optional = case.get("optional_inputs", {})
    dtype_name = case["dtype"].get("data", case["dtype"].get("qkv", "float16"))
    data_dtype = _dtype(dtype_name)
    gate_dtype = _dtype(case["dtype"].get("g", dtype_name))
    B, H_k, H_v = shape["B"], shape["H_k"], shape["H_v"]
    T, K, V = shape["T"], shape["K"], shape["V"]
    generator = torch.Generator().manual_seed(20260717)
    q = torch.randn(B, H_k, T, K, dtype=data_dtype, generator=generator)
    k = torch.randn(B, H_k, T, K, dtype=data_dtype, generator=generator)
    w = torch.randn(B, H_v, T, K, dtype=data_dtype, generator=generator) * 0.05
    d_o = torch.randn(B, H_v, T, V, dtype=data_dtype, generator=generator) * 0.05
    dv = torch.randn(B, H_v, T, V, dtype=data_dtype, generator=generator) * 0.05
    g = torch.randn(B, H_v, T, dtype=gate_dtype, generator=generator) * 0.02
    gK = torch.randn(B, H_v, T, K, dtype=data_dtype, generator=generator) if _present(optional, "gK") else None
    h0 = torch.randn(B, H_v, K, V, dtype=data_dtype, generator=generator) if _present(optional, "h0") else None
    dht = torch.randn(B, H_v, K, V, dtype=data_dtype, generator=generator) if _present(optional, "dht") else None
    return q, k, w, d_o, dv, g, gK, h0, dht


def _invoke(case, tensors):
    q, k, w, d_o, dv, g, gK, h0, dht = tensors
    attrs = case["attrs"]
    optional = case.get("optional_inputs", {})
    return chunk_gated_delta_rule_bwd_dhu(
        q.npu(), k.npu(), w.npu(), d_o.npu(), dv.npu(),
        attrs.get("scale", 1.0), attrs["chunk_size"],
        g=g.npu() if _present(optional, "g") else None,
        gK=gK.npu() if gK is not None else None,
        h0=h0.npu() if h0 is not None else None,
        dht=dht.npu() if dht is not None else None,
        cu_seqlens=optional.get("cu_seqlens"),
        chunk_indices=optional.get("chunk_indices"),
        use_exp2=attrs.get("use_exp2", False),
        transpose_state_layout=attrs.get("transpose_state_layout", False),
    )


def _run_accuracy(case, tolerance):
    tensors = _inputs(case)
    q, k, w, d_o, dv, g, _, _, _ = tensors
    dh_npu, dh0_npu, dv2_npu = _invoke(case, tensors)
    torch.npu.synchronize()
    optional = case.get("optional_inputs", {})
    attrs = case["attrs"]
    dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, d_o, dv,
        cu_seqlens=optional.get("cu_seqlens"),
        chunk_indices=optional.get("chunk_indices"),
        g=g,
        scale=attrs.get("scale", 1.0),
        chunk_size=attrs["chunk_size"],
        golden_mode="fp32",
    )
    assert dh0_npu is None and dh0_ref is None
    tol = tolerance[q.dtype.__str__().split(".")[-1]]
    torch.testing.assert_close(dh_npu.cpu().float(), dh_ref.float(), **tol)
    torch.testing.assert_close(dv2_npu.cpu().float(), dv2_ref.float(), **tol)


def _run_negative(case):
    expected = case["expect"].get("message_contains", "")
    try:
        outputs = _invoke(case, _inputs(case))
        torch.npu.synchronize()
        del outputs
    except (RuntimeError, ValueError) as exc:
        if expected and expected.lower() not in str(exc).lower():
            raise AssertionError(f"expected error containing {expected!r}, got {exc!r}") from exc
        return
    raise AssertionError(f"{case['id']} did not raise")


def main():
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    manifest = _manifest()
    for case in _selected_cases(manifest):
        if "negative" in case["tags"]:
            _run_negative(case)
        elif "accuracy" in case["tags"]:
            _run_accuracy(case, manifest["tolerance"])


if __name__ == "__main__":
    main()
