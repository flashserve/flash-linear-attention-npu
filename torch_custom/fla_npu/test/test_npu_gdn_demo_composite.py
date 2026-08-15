# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch_npu


DEVICE_ID = int(os.environ.get("TEST_DEVICE_ID", 0))
EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "flash_gated_delta_rule.py"
USE_TORCH_L2NORM = os.environ.get("GDN_TEST_TORCH_L2NORM", "0") == "1"


def load_example_module():
    spec = importlib.util.spec_from_file_location("gdn_demo_composite_test_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if USE_TORCH_L2NORM:
        def torch_l2norm_fwd(x, eps=1e-6, output_dtype=None):
            x_float = x.float()
            rstd = torch.rsqrt(torch.sum(x_float * x_float, dim=-1) + eps)
            output = (x_float * rstd.unsqueeze(-1)).to(output_dtype or x.dtype)
            return output, rstd

        def torch_l2norm_bwd(y, rstd, dy, eps=1e-6):
            y_float = y.float()
            dy_float = dy.float()
            projection = torch.sum(dy_float * y_float, dim=-1, keepdim=True)
            dx = (dy_float - projection * y_float) * rstd.float().unsqueeze(-1)
            return dx.to(y.dtype)

        module.l2norm_fwd = torch_l2norm_fwd
        module.l2norm_bwd = torch_l2norm_bwd
    return module


def assert_bit_exact(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    expected_cpu = expected.detach().cpu()
    actual_cpu = actual.detach().cpu()
    if torch.equal(expected_cpu, actual_cpu):
        return
    max_abs = float((expected_cpu.float() - actual_cpu.float()).abs().max())
    raise AssertionError(f"{name} is not bit exact; max_abs={max_abs:.6e}")


def run_path(module, state_dict, hidden_states, cu_seqlens, *, use_composite_core: bool):
    model = module.DemoGatedDeltaNet(
        hidden_size=256,
        num_value_heads=2,
        num_key_heads=2,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
        chunk_size=64,
        use_composite_core=use_composite_core,
    ).npu().to(torch.bfloat16)
    model.load_state_dict(state_dict)
    x = hidden_states.detach().clone().requires_grad_(True)
    output = model(x, cu_seqlens=cu_seqlens)
    loss = output.float().square().mean()
    loss.backward()
    torch.npu.synchronize()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    return output.detach(), x.grad.detach(), gradients


def run_case(module, name: str, cu_seqlens: torch.Tensor | None) -> None:
    torch.manual_seed(20260724)
    reference = module.DemoGatedDeltaNet(
        hidden_size=256,
        num_value_heads=2,
        num_key_heads=2,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
        chunk_size=64,
        use_composite_core=False,
    ).npu().to(torch.bfloat16)
    state_dict = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    hidden_states = (torch.randn(1, 128, 256, dtype=torch.bfloat16) * 0.02).npu()

    legacy = run_path(module, state_dict, hidden_states, cu_seqlens, use_composite_core=False)
    composite = run_path(module, state_dict, hidden_states, cu_seqlens, use_composite_core=True)
    assert_bit_exact(f"{name} output", legacy[0], composite[0])
    assert_bit_exact(f"{name} input gradient", legacy[1], composite[1])
    if legacy[2].keys() != composite[2].keys():
        raise AssertionError(f"{name} parameter gradient sets differ")
    for parameter_name in legacy[2]:
        assert_bit_exact(
            f"{name} gradient {parameter_name}",
            legacy[2][parameter_name],
            composite[2][parameter_name],
        )
    print(f"{name}: PASS")


def main() -> None:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    module = load_example_module()
    run_case(module, "dense_bf16", None)
    run_case(module, "varlen_equal_bf16", torch.tensor([0, 64, 128], dtype=torch.int64, device="npu"))
    print("GDN Demo composite A/B: 2/2 PASS")


if __name__ == "__main__":
    main()
