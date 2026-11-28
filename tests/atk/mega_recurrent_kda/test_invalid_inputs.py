"""Run against the installed A5 wheel with ASCEND_LAUNCH_BLOCKING=1."""

import pytest
import torch

pytest.importorskip("atk")
pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")

from executor_mega_recurrent_kda import build_inputs, to_device


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate_slot",
        "slot_outside_pool",
        "accepted_overflow",
        "invalid_prefix",
        "wrong_index_dtype",
        "inner_stride",
        "invalid_scale",
        "invalid_lower_bound",
        "invalid_epsilon",
        "requires_grad",
    ],
)
def test_invalid_inputs_preserve_state(invalid):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    host = build_inputs(dict(lengths=[2, 1], heads=1, layout="TH", name="invalid", conv="shared", seed=0))
    if invalid == "duplicate_slot":
        host["state_indices"][1] = host["state_indices"][0]
    elif invalid == "slot_outside_pool":
        host["state_indices"][0] = host["replay_u"].shape[0]
    elif invalid == "accepted_overflow":
        host["conv_num_accepted_tokens"][0] = host["conv_state"].shape[1]
    elif invalid == "invalid_prefix":
        host["cu_seqlens"][-1] += 1
    elif invalid == "wrong_index_dtype":
        host["state_indices"] = host["state_indices"].long()
    elif invalid == "invalid_scale":
        host["scale"] = float("inf")
    elif invalid == "invalid_lower_bound":
        host["lower_bound"] = -6.0
    elif invalid == "invalid_epsilon":
        host["rms_norm_eps"] = 0

    device = torch.device("npu", torch.npu.current_device())
    args = {k: to_device(v, device) for k, v in host.items()}
    args["a_log"] = args.pop("A_log")
    if invalid == "inner_stride":
        pool = args["conv_state"]
        args["conv_state"] = pool.transpose(1, 2).contiguous().transpose(1, 2)
    elif invalid == "requires_grad":
        args["replay_u"].requires_grad_()

    names = ("conv_state", "checkpoint_state", "replay_u", "replay_k", "replay_decay")
    before = {k: args[k].detach().cpu().clone() for k in names}
    with pytest.raises((ValueError, RuntimeError)):
        npu_mega_recurrent_kda(**args)
        torch.npu.synchronize()
    assert all(torch.equal(args[k].detach().cpu(), old) for k, old in before.items())
