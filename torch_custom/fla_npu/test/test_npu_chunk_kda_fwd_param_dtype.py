"""Native BF16 gate parameters must match equal-valued FP32 parameters."""
import itertools

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import _stable, _aclnn_ctypes


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "NTD", "TND"])
@pytest.mark.parametrize("dtypes", list(itertools.product([torch.bfloat16, torch.float32], repeat=4)))
def test_forward_parameter_dtypes(layout, dtypes):
    torch.manual_seed(993)
    device = torch.device("npu", torch.npu.current_device())
    packed = layout in ("NTD", "TND")
    tm = layout in ("BSND", "TND")
    shape = (65, 3, 128) if tm else (3, 65, 128)
    if not packed:
        shape = (2, *shape)
    q, k, v = [(torch.randn(shape) * 0.05).to(device, torch.bfloat16) for _ in range(3)]
    g = (torch.randn(shape) * 0.1).to(device, dtypes[0])
    beta = torch.sigmoid(torch.randn(shape[:-1])).to(device, dtypes[1])
    a_log = (torch.randn(3) * 0.2 - 0.5).to(device, dtypes[2])
    bias = (torch.randn(384) * 0.1).to(device, dtypes[3])
    args = dict(layout=layout, cu_seqlens=[0, 1, 65] if packed else None,
                safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
                use_qk_l2norm_in_kernel=True, use_exp2=True)
    for backend in (_stable, _aclnn_ctypes):
        for name in ("npu_chunk_kda_fwd", "npu_chunk_kda_fwd_prepare"):
            op = getattr(backend, name)
            options = dict(args)
            if name == "npu_chunk_kda_fwd":
                options.update(disable_recompute=True, return_intermediate_states=True,
                               output_final_state=True)
            control = op(q, k, v, g, beta, 128**-0.5,
                         A_log=a_log.float(), dt_bias=bias.float(), **options)
            actual = op(q, k, v, g, beta, 128**-0.5,
                        A_log=a_log, dt_bias=bias, **options)
            torch.npu.synchronize()
            assert len(actual) == len(control)
            for got, expected in zip(actual, control):
                if expected is None:
                    assert got is None
                else:
                    assert torch.isfinite(got).all()
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["none", "forward", "recompute", "save"])
@pytest.mark.parametrize("gate_mode", ["safe", "softplus", "precomputed"])
def test_optional_parameters_and_save_modes(mode, gate_mode):
    torch.manual_seed(994)
    device = torch.device("npu", torch.npu.current_device())
    shape = (1, 3, 64, 128)
    q, k, v = [(torch.randn(shape) * 0.02).to(device, torch.bfloat16) for _ in range(3)]
    g = (-torch.rand(shape) * 0.05).to(device, torch.bfloat16)
    beta = torch.randn(shape[:-1]).to(device, torch.bfloat16)
    a = None if gate_mode == "precomputed" else torch.randn(3).to(device, torch.bfloat16)
    args = dict(layout="BNSD", use_gate_in_kernel=a is not None,
                safe_gate=gate_mode == "safe", lower_bound=-5.0,
                use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True,
                use_exp2=False, backward_mode=mode)
    for backend in (_stable, _aclnn_ctypes):
        op = backend.npu_chunk_kda_fwd_prepare
        control = op(q, k, v, g, beta, 128**-0.5,
                     A_log=None if a is None else a.float(), dt_bias=None, **args)
        actual = op(q, k, v, g, beta, 128**-0.5, A_log=a, dt_bias=None, **args)
        for got, expected in zip(actual, control):
            if expected is None:
                assert got is None
            else:
                torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_bf16_parameters_select_v2_on_small_default_case():
    torch.manual_seed(995)
    device = torch.device("npu", torch.npu.current_device())
    shape = (1, 3, 65, 128)
    q, k, v, g = [(torch.randn(shape) * 0.02).to(device, torch.bfloat16) for _ in range(4)]
    beta = torch.sigmoid(torch.randn(shape[:-1])).to(device, torch.bfloat16)
    a = torch.randn(3).to(device, torch.bfloat16)
    bias = torch.randn(384).to(device, torch.bfloat16)
    options = dict(layout="BNSD", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0)
    for backend in (_stable, _aclnn_ctypes):
        control = backend.npu_chunk_kda_fwd_prepare(
            q, k, v, g, beta, 128**-0.5, A_log=a.float(), dt_bias=bias.float(), **options)
        actual = backend.npu_chunk_kda_fwd(
            q, k, v, g, beta, 128**-0.5, A_log=a, dt_bias=bias,
            disable_recompute=True, return_intermediate_states=True, **options)
        for total_index, prepare_index in zip(range(2, 9), range(7)):
            torch.testing.assert_close(actual[total_index], control[prepare_index], rtol=0, atol=0)
        for name in ("A_log", "dt_bias"):
            bad = dict(A_log=a, dt_bias=bias)
            bad[name] = bad[name].half()
            with pytest.raises(RuntimeError):
                backend.npu_chunk_kda_fwd(q, k, v, g, beta, 128**-0.5, **bad, **options)
