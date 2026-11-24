"""Device plan and in-place output regressions."""
from bisect import bisect_right
from itertools import accumulate

import pytest
import torch

pytest.importorskip("torch_npu")
from executor_mega_recurrent_kda import build_inputs, to_device

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("lengths", [[1], [16, 48], [63, 2], [1] * 65, [193, 3, 64]])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sequence_plan(lengths, dtype):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda_metadata

    ends = [0, *accumulate(lengths)]
    prefix = torch.tensor(ends, dtype=dtype, device="npu")
    expected = [int(any(a // 64 != (b - 1) // 64 for a, b in zip(ends, ends[1:]))), max(lengths)]
    expected.extend(bisect_right(ends, row) - 1 for row in range(0, ends[-1], 16))
    assert npu_mega_recurrent_kda_metadata(prefix, ends[-1]).cpu().tolist() == expected


@pytest.mark.parametrize("lengths,layout,dtype", [
    ([4] * 30, "TH", torch.int32),
    ([16, 48, 16], "TH", torch.int64),
    ([63, 2, 55], "TH", torch.int64),
    ([1, 193, 3], "TH", torch.int32),
    ([3] * 22, "BSH", torch.int32),
    ([4] * 40, "BSH", torch.int32),
])
def test_consumer_matches_legacy(lengths, layout, dtype):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda, npu_mega_recurrent_kda_metadata

    spec = dict(name="metadata", lengths=lengths, heads=2, layout=layout, conv="flat", seed=19)
    host = build_inputs(spec)
    dev = torch.device("npu", torch.npu.current_device())
    args = {key: to_device(value, dev) for key, value in host.items()}
    args["a_log"] = args.pop("A_log")
    if layout == "TH":
        args["cu_seqlens"] = args["cu_seqlens"].to(dtype)
    mutable = ("conv_state", "replay_u", "replay_k", "replay_decay")
    baseline_args = dict(args, **{key: args[key].clone() for key in mutable})
    checkpoint = args["checkpoint_state"].clone()
    plan = npu_mega_recurrent_kda_metadata(args["cu_seqlens"], sum(lengths)) if layout == "TH" else None
    for _ in range(2):
        expected = npu_mega_recurrent_kda(**baseline_args)
        actual = npu_mega_recurrent_kda(**args, metadata=plan, use_device_metadata=True)
        for got, want in zip(actual, expected):
            assert torch.equal(got, want)
        assert all(t.data_ptr() == args[k].data_ptr() for t, k in zip(actual[1:], mutable))
        assert torch.equal(checkpoint, args["checkpoint_state"])
