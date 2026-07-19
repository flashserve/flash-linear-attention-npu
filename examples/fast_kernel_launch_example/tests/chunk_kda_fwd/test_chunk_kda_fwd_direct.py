import json
from pathlib import Path

import pytest
import torch
import torch_npu

import ascend_ops  # noqa: F401
from fla_npu.ops import ascendc as fla_ascendc


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_kda_fwd_direct_matches_aclnn(dtype):
    manifest = json.loads((ROOT / "tests/op_cases/chunk_kda_fwd.json").read_text(encoding="utf-8"))
    case = next(item for item in manifest["cases"] if item["id"] == "chunk_kda_fwd_main_accuracy")
    shape = case["shape"]
    torch.manual_seed(case["seed"])
    device = torch.device("npu:0")
    batch, h_k, h_v = shape["B"], shape["H_k"], shape["H_v"]
    seqlen, kdim, vdim = shape["T"], shape["K"], shape["V"]
    chunk_size = shape["chunk_size"]
    scale = case["attrs"]["scale"]

    q = (torch.randn(batch, h_k, seqlen, kdim) * 0.08).to(dtype).to(device)
    k = (torch.randn(batch, h_k, seqlen, kdim) * 0.08).to(dtype).to(device)
    v = (torch.randn(batch, h_v, seqlen, vdim) * 0.08).to(dtype).to(device)
    gate_steps = -(torch.rand(batch, h_v, seqlen // chunk_size, chunk_size, kdim) * 0.01 + 0.002)
    gk = gate_steps.cumsum(dim=3).reshape(batch, h_v, seqlen, kdim).to(device)
    beta = torch.sigmoid(torch.randn(batch, h_v, seqlen)).to(device)
    initial_state = (torch.randn(batch, h_v, kdim, vdim) * 0.02).float().to(device)

    direct = torch.ops.ascend_ops.chunk_kda_fwd_direct(
        q, k, v, gk, beta, scale, chunk_size,
        initial_state=initial_state, output_final_state=True,
    )
    aclnn = fla_ascendc.chunk_kda_fwd(
        q, k, v, gk, beta, scale, chunk_size, layout="BNSD",
        initial_state=initial_state, output_final_state=True, return_intermediate=True,
    )
    for name, actual, reference in zip(
        (
            "o", "final_state", "g", "Aqk", "Akk", "w", "u", "qg", "kg",
            "v_new", "h", "initial_state_out",
        ),
        direct,
        aclnn,
    ):
        assert torch.isfinite(actual).all().item(), f"direct {name} contains NaN or Inf"
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
