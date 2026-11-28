"""Compare row-indexed A3 verification with physical-page INT8 MegaKDA."""
import importlib
import sys
from itertools import accumulate
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))
from mega_recurrent_kda_reference import _make_case


@pytest.mark.parametrize("lengths", [(4, 4), (1, 3)])
def test_verify_a3_request_rows(lengths):
    if not torch.npu.is_available() or not torch.npu.get_device_name().startswith(("Ascend910_93", "Ascend910C")):
        pytest.skip("requires Atlas A3")
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    provider = npu_mega_recurrent_kda.__module__.split(".ops.ascendc")[0]
    stable = importlib.import_module(provider + ".ops.ascendc._stable")
    verify = stable._op("npu_mega_recurrent_kda_verify_a3")
    args, tokens, _ = _make_case(2, 4, 1, block_num=5, accepted=[2, 1], seed=913)
    weights = args["qkv_projection_weight"].float().T.contiguous()
    scales = (weights.abs().amax(dim=0) / 127).bfloat16()
    codes = (weights / scales).round().clamp(-127, 127).to(torch.int8)
    args["qkv_projection_weight"] = codes.reshape(448, 16, 12, 32).permute(2, 0, 1, 3).contiguous()
    args["qkv_projection_weight_scale"] = scales
    slots = [3, 1]
    args["state_indices"] = torch.tensor(slots, dtype=torch.int32)
    args = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in args.items()}
    # Model graphs can retain padded input rows after the last real token.
    starts = [0, *accumulate(lengths)]
    dense = torch.stack([tokens[start:start + 4] for start in starts[:-1]]).npu()
    source_conv = args["conv_state"].clone()
    checkpoint = args["checkpoint_state"].clone()
    reference, *_ = npu_mega_recurrent_kda(hidden_states=dense, rms_norm_eps=1e-5, **args)
    names = ("conv_state", "replay_u", "replay_k", "replay_decay")
    records = {k: torch.empty_like(args[k][:2]) for k in names}
    inputs = dict(args, **records)
    weight_names = (
        "qkv_projection_weight", "decay_projection_a_weight", "decay_projection_b_weight",
        "beta_projection_weight", "output_gate_projection_weight", "output_norm_weight",
        "output_projection_weight", "conv1d_weight", "conv_state", "checkpoint_state",
        "replay_u", "replay_k", "replay_decay", "state_indices", "conv_num_accepted_tokens", "A_log", "dt_bias",
    )
    hidden = tokens.npu()
    out = torch.full_like(hidden, float("nan"))
    flag = torch.zeros(1, dtype=torch.int32, device=hidden.device)
    expected_conv = source_conv.clone()
    verify(hidden, *(inputs[k] for k in weight_names), None, None, scales.npu(),
           args["scale"], args["lower_bound"], 1e-5, source_conv, torch.tensor(starts, dtype=torch.int32).npu(),
           flag, out, stable._current_stream_ptr())
    torch.npu.synchronize()
    assert flag.item() == 1
    torch.testing.assert_close(source_conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(args["checkpoint_state"], checkpoint, atol=0, rtol=0)
    for row, (slot, size) in enumerate(zip(slots, lengths)):
        torch.testing.assert_close(out[starts[row]:starts[row + 1]], reference[row, :size], atol=0, rtol=0)
        for name in names:
            torch.testing.assert_close(records[name][row], args[name][slot], atol=0, rtol=0)
    assert torch.count_nonzero(out[starts[-1]:]) == 0
