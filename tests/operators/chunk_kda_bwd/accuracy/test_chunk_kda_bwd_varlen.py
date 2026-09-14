"""Packed-varlen regressions for the fused KDA backward operator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F


CASE_FILE = Path(__file__).resolve().parents[3] / "op_cases/chunk_kda_bwd.json"
with CASE_FILE.open(encoding="utf-8") as file:
    MANIFEST = json.load(file)
CASES = MANIFEST["cases"]
A2_CASES = [case for case in CASES if "a2" in case["tags"]]
A5_CASES = [case for case in CASES if "a5" in case["tags"]]

CHUNK_SIZE = 64
KEY_DIM = 128
VALUE_DIM = 128
OUTPUT_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
DEVICE_ID = int(os.getenv("TEST_DEVICE_ID", "0"))


def test_case_manifest_contract():
    assert MANIFEST["op"] == "chunk_kda_bwd"
    assert MANIFEST["implementation"] == "ascendc"
    assert MANIFEST["capability"] == {
        "run_on": ["ascendc"],
        "soc": ["ascend910b", "ascend950"],
        "layout": ["BNSD", "NTD"],
    }
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert all(
        case["shape"]["key_dim"] == KEY_DIM
        and case["shape"]["value_dim"] == VALUE_DIM
        for case in CASES
    )

    issue_case = next(
        case for case in CASES if case["id"] == "a2_issue_544_partial_tail_15"
    )
    assert issue_case["shape"] == {
        "seq_lengths": [640, 783],
        "heads": 96,
        "key_dim": 128,
        "value_dim": 128,
    }
    assert {
        case["shape"]["seq_lengths"][1] % CHUNK_SIZE for case in A2_CASES
    } == {0, 7, 8, 9, 15, 16, 23, 24, 25, 31, 32}
    assert {
        case["id"]: case["shape"]["seq_lengths"] for case in A5_CASES
    } == {
        "a5_varlen_lower_a_padding_8": [64, 72],
        "a5_varlen_lower_a_padding_24": [64, 88],
        "a5_varlen_lower_a_padding_40": [64, 104],
        "a5_varlen_lower_a_padding_56": [64, 120],
        "a5_issue_544_original_tail_15": [640, 783],
    }


def _is_ascend910b() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available() and str(
            torch.npu.get_device_name(DEVICE_ID)
        ).startswith("Ascend910B")
    except (AttributeError, ImportError, RuntimeError):
        return False


def _is_ascend950() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available() and "950" in str(
            torch.npu.get_device_name(DEVICE_ID)
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


def _bsnd_to_bnsd(tensor):
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsh_to_bhs(tensor):
    return tensor.permute(0, 2, 1).contiguous()


def _bsnd_to_ntd(tensor):
    return tensor.squeeze(0).permute(1, 0, 2).contiguous()


def _bsh_to_nt(tensor):
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _bnsd_to_bsnd(tensor):
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bhs_to_bsh(tensor):
    return tensor.permute(0, 2, 1).contiguous()


def _ntd_to_bsnd(tensor):
    return tensor.permute(1, 0, 2).unsqueeze(0).contiguous()


def _nt_to_bsh(tensor):
    return tensor.transpose(0, 1).unsqueeze(0).contiguous()


def _make_inputs(total: int, heads: int, seed: int, device):
    torch.manual_seed(seed)

    def rand(*shape, scale=1.0, dtype=torch.bfloat16):
        return torch.randn(*shape, device=device, dtype=dtype) * scale

    q = F.normalize(
        rand(1, total, heads, KEY_DIM, scale=0.1).float(), dim=-1
    ).bfloat16()
    k = F.normalize(
        rand(1, total, heads, KEY_DIM, scale=0.1).float(), dim=-1
    ).bfloat16()
    beta_raw = rand(1, total, heads, scale=0.01, dtype=torch.float32)
    return {
        "q": q,
        "k": k,
        "v": rand(1, total, heads, VALUE_DIM, scale=0.1),
        "g": rand(1, total, heads, KEY_DIM, scale=6.0),
        "beta": torch.sigmoid(beta_raw),
        "do": rand(1, total, heads, VALUE_DIM),
        "A_log": torch.full(
            (heads,), -1.0, device=device, dtype=torch.float32
        ),
        "dt_bias": torch.zeros(
            heads * KEY_DIM, device=device, dtype=torch.float32
        ),
    }


def _prepare_native_backward(inputs, start: int, end: int, cu_seqlens=None):
    from fla_npu.ops.ascendc import chunk_kda_fwd

    q, k, v, raw_g, beta, d_o = (
        inputs[name][:, start:end].contiguous()
        for name in ("q", "k", "v", "g", "beta", "do")
    )
    host_cu = (
        None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)
    )
    outputs = chunk_kda_fwd(
        q,
        k,
        v,
        raw_g,
        beta,
        KEY_DIM**-0.5,
        CHUNK_SIZE,
        layout="BSND",
        initial_state=None,
        output_final_state=False,
        cu_seqlens=host_cu,
        chunk_indices=None,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=False,
    )
    _, _, gk, aqk, akk, w, _, qg, kg, v_new, h, _ = outputs

    if host_cu is None:
        args = (
            _bsnd_to_bnsd(q),
            _bsnd_to_bnsd(k),
            _bsnd_to_bnsd(v),
            _bsh_to_bhs(beta),
            gk,
            aqk,
            akk,
            w,
            qg,
            kg,
            v_new,
            h,
            _bsnd_to_bnsd(d_o),
        )
        raw_g_bwd = _bsnd_to_bnsd(raw_g)
    else:
        args = (
            _bsnd_to_ntd(q),
            _bsnd_to_ntd(k),
            _bsnd_to_ntd(v),
            _bsh_to_nt(beta),
            gk.squeeze(0),
            aqk.squeeze(0),
            akk.squeeze(0),
            w.squeeze(0),
            qg.squeeze(0),
            kg.squeeze(0),
            v_new.squeeze(0),
            h.squeeze(0),
            _bsnd_to_ntd(d_o),
        )
        raw_g_bwd = _bsnd_to_ntd(raw_g)

    return args, raw_g_bwd, host_cu


def _run_backward(inputs, args, raw_g_bwd, host_cu):
    from fla_npu.ops.ascendc import chunk_kda_bwd

    dq, dk, dv, db, dg, _, _, _ = chunk_kda_bwd(
        *args,
        KEY_DIM**-0.5,
        raw_g=raw_g_bwd,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"].reshape(-1, KEY_DIM).contiguous(),
        initial_state=None,
        dht=None,
        cu_seqlens=host_cu,
        chunk_indices=None,
        chunk_size=CHUNK_SIZE,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        disable_recompute=True,
        use_exp2=True,
        state_v_first=False,
    )

    if host_cu is None:
        tensors = (
            _bnsd_to_bsnd(dq),
            _bnsd_to_bsnd(dk),
            _bnsd_to_bsnd(dv),
            _bnsd_to_bsnd(dg),
            _bhs_to_bsh(db),
        )
    else:
        tensors = (
            _ntd_to_bsnd(dq),
            _ntd_to_bsnd(dk),
            _ntd_to_bsnd(dv),
            _ntd_to_bsnd(dg),
            _nt_to_bsh(db),
        )
    return dict(zip(OUTPUT_NAMES, tensors))


def _run_native(inputs, start: int, end: int, cu_seqlens=None):
    args, raw_g_bwd, host_cu = _prepare_native_backward(
        inputs, start, end, cu_seqlens
    )
    return _run_backward(inputs, args, raw_g_bwd, host_cu)


@pytest.mark.npu
@pytest.mark.skipif(not _is_ascend910b(), reason="requires an Ascend 910B NPU")
@pytest.mark.parametrize("case", A2_CASES, ids=lambda case: case["id"])
@torch.inference_mode()
def test_packed_varlen_matches_independent_sequences(case):
    torch.npu.set_device(DEVICE_ID)
    device = torch.device(f"npu:{DEVICE_ID}")
    first, second = case["shape"]["seq_lengths"]
    total = first + second
    inputs = _make_inputs(total, case["shape"]["heads"], case["seed"], device)

    packed = _run_native(inputs, 0, total, (0, first, total))
    independent = (
        _run_native(inputs, 0, first),
        _run_native(inputs, first, total),
    )
    torch.npu.synchronize()

    for name in OUTPUT_NAMES:
        expected = torch.cat(
            (independent[0][name], independent[1][name]), dim=1
        )
        actual_cpu = packed[name].float().cpu()
        expected_cpu = expected.float().cpu()
        assert torch.isfinite(actual_cpu).all(), f"{name} contains non-finite values"
        assert torch.isfinite(expected_cpu).all(), (
            f"independent {name} contains non-finite values"
        )
        torch.testing.assert_close(
            actual_cpu,
            expected_cpu,
            rtol=2e-4,
            atol=2e-4,
            msg=f"{case['id']} {name} mismatch",
        )


@pytest.mark.npu
@pytest.mark.skipif(not _is_ascend950(), reason="requires an Ascend 950 NPU")
@pytest.mark.parametrize("case", A5_CASES, ids=lambda case: case["id"])
@torch.inference_mode()
def test_a5_varlen_lower_a_padding_with_poisoned_workspace(case):
    from fla_npu.ops.ascendc import _aclnn_ctypes as ascendc_ctypes

    torch.npu.set_device(DEVICE_ID)
    device = torch.device(f"npu:{DEVICE_ID}")
    first, second = case["shape"]["seq_lengths"]
    total = first + second
    inputs = _make_inputs(total, case["shape"]["heads"], case["seed"], device)
    packed_args, packed_raw_g, host_cu = _prepare_native_backward(
        inputs, 0, total, (0, first, total)
    )

    # Use the very same saved forward tensors for the independent dense baseline.
    independent = []
    chunk_begin = 0
    for begin, end in ((0, first), (first, total)):
        chunk_count = (end - begin + CHUNK_SIZE - 1) // CHUNK_SIZE
        # Argument 11 is h (chunk-major); all other arguments are token-major.
        dense_args = tuple(
            tensor.narrow(
                0 if index == 11 else 1,
                chunk_begin if index == 11 else begin,
                chunk_count if index == 11 else end - begin,
            ).unsqueeze(0).contiguous()
            for index, tensor in enumerate(packed_args)
        )
        dense_raw_g = (
            packed_raw_g.narrow(1, begin, end - begin).unsqueeze(0).contiguous()
        )
        independent.append(_run_backward(inputs, dense_args, dense_raw_g, None))
        chunk_begin += chunk_count

    original_empty = torch.empty
    poisoned_workspaces = []

    def poison_uint8_workspace(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.dtype == torch.uint8 and tensor.device.type == "npu":
            tensor.fill_(0xFF)
            poisoned_workspaces.append(tensor)
        return tensor

    # Only the packed backward is patched: forward and dense baseline remain native.
    with mock.patch.object(
        torch.npu, "get_device_name", return_value="Ascend910_93"
    ), mock.patch.object(
        torch, "empty", side_effect=poison_uint8_workspace
    ), mock.patch.object(
        ascendc_ctypes, "_call_aclnn", wraps=ascendc_ctypes._call_aclnn
    ) as aclnn_call:
        packed = _run_backward(inputs, packed_args, packed_raw_g, host_cu)

    assert aclnn_call.call_count == 1
    assert aclnn_call.call_args.args[0] == "aclnnChunkKdaBwd"
    assert len(poisoned_workspaces) == 1
    torch.npu.synchronize()

    for name in OUTPUT_NAMES:
        expected = torch.cat((independent[0][name], independent[1][name]), dim=1)
        actual_cpu = packed[name].float().cpu()
        expected_cpu = expected.float().cpu()
        assert torch.isfinite(actual_cpu).all(), f"{name} contains non-finite values"
        assert torch.isfinite(expected_cpu).all(), (
            f"independent {name} contains non-finite values"
        )
        torch.testing.assert_close(
            actual_cpu,
            expected_cpu,
            rtol=2e-4,
            atol=2e-4,
            msg=f"{case['id']} {name} mismatch",
        )
