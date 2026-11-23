"""Short projection and cross-tile recurrence regressions for the public A5 kernel."""
import pytest
import torch

pytest.importorskip("torch_npu")
from executor_mega_recurrent_kda import build_inputs, mega_recurrent_kda_golden, to_device


@pytest.mark.parametrize("lengths,heads,layout,output,atol,rtol", [
    ([1], 7, "BSH", 4, 1.53e-5, 9.77e-4),
    ([63, 63], 24, "TH", 3, 1.95e-3, 1.95e-3),
    ([3] * 40, 8, "BSH", 3, 1.95e-3, 1.95e-3),
    ([3] * 22, 8, "TH", 4, 1.53e-5, 9.77e-4),
    ([63, 65, 62, 67, 64, 3, 61], 3, "TH", 2, 1.53e-5, 9.77e-4),
])
def test_projection_and_history_handoff(lengths, heads, layout, output, atol, rtol):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    spec = dict(name="regression", lengths=lengths, heads=heads, layout=layout, conv="flat", seed=0)
    host = build_inputs(spec)
    device = torch.device("npu", torch.npu.current_device())
    args = {k: to_device(v, device) for k, v in host.items()}
    args["a_log"] = args.pop("A_log")
    mutable = ("conv_state", "replay_u", "replay_k", "replay_decay")
    initial = {k: args[k].clone() for k in mutable}
    expected = mega_recurrent_kda_golden(**host)[output]
    slots = host["state_indices"].long()
    def active_rows(tensor):
        return torch.cat([tensor[slot, :, :length].reshape(-1) for slot, length in zip(slots, lengths)])

    expected = active_rows(expected).float()
    previous = None
    for _ in range(3):
        for k, value in initial.items():
            args[k].copy_(value)
        result = active_rows(npu_mega_recurrent_kda(**args)[output].cpu())
        assert torch.isfinite(result).all()
        delta = (result.float() - expected).abs()
        assert (delta > atol + rtol * expected.abs()).sum().item() <= delta.numel() * 0.001
        if previous is not None:
            assert torch.equal(result, previous)
        previous = result
