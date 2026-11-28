"""Precision regressions for the public MegaRecurrentKda operator."""

import pytest
import torch

pytest.importorskip("torch_npu")
from executor_mega_recurrent_kda import build_inputs, device_inputs, expected_outputs, to_device


@pytest.mark.parametrize(
    "lengths,heads,layout,output,atol,rtol",
    [
        ([1], 7, "BSH", 4, 1.53e-5, 9.77e-4),
        ([63, 63], 24, "TH", 3, 1.95e-3, 1.95e-3),
        ([3] * 40, 8, "BSH", 3, 1.95e-3, 1.95e-3),
        ([3] * 22, 8, "TH", 4, 1.53e-5, 9.77e-4),
        ([63, 65, 62, 67, 64, 3, 61], 3, "TH", 2, 1.53e-5, 9.77e-4),
        ([31] * 8 + [8, 1], 8, "TH", 0, 1.56e-2, 1.56e-2),
        ([16, 48, 4], 8, "TH", 0, 1.56e-2, 1.56e-2),
    ],
)
def test_projection_and_history_handoff(lengths, heads, layout, output, atol, rtol):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    spec = dict(name="regression", lengths=lengths, heads=heads, layout=layout, conv="flat", seed=0)
    host = build_inputs(spec)
    device = torch.device("npu", torch.npu.current_device())
    args = {k: to_device(v, device) for k, v in host.items()}
    args["a_log"] = args.pop("A_log")
    mutable = ("conv_state", "replay_u", "replay_k", "replay_decay")
    initial = {k: args[k].clone() for k in mutable}
    expected = expected_outputs(host)[output]
    slots = host["state_indices"].long()

    def active_rows(tensor):
        if output == 0:
            return tensor.reshape(-1)
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


@pytest.mark.parametrize("lengths,layout", [([4, 4], "BSH"), ([63, 4], "TH")])
def test_arch22_zero_correction(lengths, layout):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    chip = torch.npu.get_device_name()
    if not chip.startswith(("Ascend910B", "Ascend910_93")):
        pytest.skip("Checks the Atlas A2/A3 replay representation")
    host = build_inputs(dict(name="zero", lengths=lengths, heads=3, layout=layout, conv="flat", seed=0))
    host["hidden_states"].fill_(-0.0)
    host["conv_state"].zero_()
    host["checkpoint_state"].fill_(-0.0)
    device = torch.device("npu", torch.npu.current_device())
    args = {k: to_device(v, device) for k, v in host.items()}
    args["a_log"] = args.pop("A_log")
    replay_u = npu_mega_recurrent_kda(**args)[2].cpu()
    for slot, length in zip(host["state_indices"].tolist(), lengths):
        active = replay_u[slot, :, :length].contiguous()
        assert torch.count_nonzero(active.view(torch.int32)).item() == 0


@pytest.mark.parametrize("hidden,lengths,layout", [
    (4096, [4, 4], "BSH"), (4097, [4, 4], "BSH"), (17, [63, 2], "TH"), (1, [2], "BSH"),
    (4112, [63, 2], "TH"), (7168, [2], "BSH"),
])
def test_hidden_width(hidden, lengths, layout):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    host = build_inputs(dict(name="width", hidden=hidden, lengths=lengths, heads=2, layout=layout, conv="flat", seed=17))
    args = device_inputs(host, torch.device("npu", torch.npu.current_device()))
    outputs = npu_mega_recurrent_kda(**args)
    assert outputs[0].shape == args["hidden_states"].shape
    tolerance = [(0.0156, 0.0156), (0.0156, 0.0156), (1.53e-5, 9.77e-4),
                 (1.95e-3, 1.95e-3), (1.53e-5, 9.77e-4)]
    for i, (got, want) in enumerate(zip(outputs, expected_outputs(host))):
        got = got.cpu().float()
        if i:
            assert got.shape == want.shape
            slots = host["state_indices"].tolist()
            if i > 1:
                got, want = got.transpose(1, 2), want.transpose(1, 2)
            active = torch.zeros(got.shape[:2], dtype=torch.bool)
            for slot, n in zip(slots, lengths):
                active[slot, :n + (2 if i == 1 else 0)] = True
            assert torch.equal(got[~active], want[~active])
            got, want = got[active], want[active]
        atol, rtol = tolerance[i]
        delta = (got - want.float()).abs()
        assert torch.isfinite(delta).all()
        assert (delta > atol + rtol * want.abs()).sum().item() <= got.numel() * 0.001, i
