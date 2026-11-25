"""Native FP32 checkpoint, resume and legacy-output contracts on Ascend950."""

import importlib.util
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_kda_fwd  # noqa: E402


@pytest.fixture(autouse=True)
def require_ascend950():
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("Native internal KDA checkpoints currently require Ascend950")


def _call(
    inputs,
    starts,
    initial,
    *,
    offsets=None,
    checkpoint=None,
    state_v_first=True,
    use_qk_l2norm_in_kernel=True,
    save_prepared=False,
    output_final_state=True,
):
    q, k, v, raw_gate, beta, a_log, bias = inputs
    return chunk_kda_fwd(
        q,
        k,
        v,
        raw_gate,
        beta,
        128**-0.5,
        64,
        cu_seqlens=starts,
        initial_state=initial,
        output_final_state=output_final_state,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        state_v_first=state_v_first,
        checkpoint_offsets=offsets,
        checkpoint_state=checkpoint,
        disable_recompute=save_prepared,
        return_intermediate_states=save_prepared,
    )


def _fwd_h_fp64_reference(*args):
    path = (
        Path(__file__).resolve().parents[3]
        / "tests/atk/chunk_kda_fwd/checkpoint_reference.py"
    )
    spec = importlib.util.spec_from_file_location("_kda_cpu_checkpoint_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fwd_h_fp64_reference(*args)


def _relative_rmse(actual, expected, limit=0.005):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    ratio = (
        actual - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    assert ratio.item() < limit, ratio.item()


@pytest.mark.npu
@pytest.mark.parametrize("state_v_first", [True, False], ids=["VK", "KV"])
@pytest.mark.parametrize("cold", [False, True, None], ids=["warm", "zeros", "absent"])
@pytest.mark.parametrize(
    "heads,starts,offsets_host",
    [
        pytest.param(2, [0, 65, 258, 275], [64, 128, 0], id="short-H2"),
        pytest.param(12, [0, 1537, 4465, 4482], [1536, 1536, 0], id="production-H12"),
        pytest.param(128, [0, 65, 258, 275], [64, 128, 0], id="head-round-reuse-H128"),
    ],
)
def test_native_checkpoint_and_resume(state_v_first, cold, heads, starts, offsets_host):
    torch.npu.set_device(0)
    torch.manual_seed(17)
    lens = [b - a for a, b in zip(starts, starts[1:])]
    offsets = torch.tensor(offsets_host, dtype=torch.int32, device="npu")
    inputs = [
        torch.randn(1, starts[-1], heads, 128, dtype=torch.bfloat16, device="npu")
        for _ in range(4)
    ]
    inputs += [
        torch.rand(1, starts[-1], heads, device="npu"),
        torch.zeros(heads, device="npu"),
        torch.zeros(heads * 128, device="npu"),
    ]
    initial = torch.randn(3, heads, 128, 128, dtype=torch.float32, device="npu")
    if cold:
        initial.zero_()
    cp = torch.full_like(initial, 19.0)
    saved_inputs = [t.clone() for t in inputs]
    base = _call(
        inputs,
        starts,
        None if cold is None else initial.clone(),
        state_v_first=state_v_first,
    )
    got = _call(
        inputs,
        starts,
        None if cold is None else initial.clone(),
        offsets=offsets,
        checkpoint=cp,
        state_v_first=state_v_first,
    )
    # Same composed algorithm with optional export: preserve default outputs.
    torch.testing.assert_close(got[0], base[0], rtol=0, atol=0)
    torch.testing.assert_close(got[1], base[1], rtol=0, atol=0)
    torch.testing.assert_close(cp[2], torch.full_like(cp[2], 19.0), rtol=0, atol=0)
    for actual, expected in zip(inputs, saved_inputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for row, offset in enumerate(offsets_host):
        if not offset:
            continue
        start, length = starts[row], lens[row]
        prefix = [
            t[:, start : start + offset].contiguous() if i < 5 else t
            for i, t in enumerate(inputs)
        ]
        prefix_result = _call(
            prefix,
            [0, offset],
            None if cold is None else initial[row : row + 1].clone(),
            state_v_first=state_v_first,
        )
        torch.testing.assert_close(cp[row : row + 1], prefix_result[1], rtol=0, atol=0)
        tail = [
            t[:, start + offset : start + length].contiguous() if i < 5 else t
            for i, t in enumerate(inputs)
        ]
        resumed = _call(
            tail,
            [0, length - offset],
            cp[row : row + 1].clone(),
            state_v_first=state_v_first,
        )
        _relative_rmse(resumed[0], got[0][:, start + offset : start + length])
        _relative_rmse(resumed[1], got[1][row : row + 1])


@pytest.mark.npu
def test_checkpoint_rejects_alias():
    x = torch.zeros(1, 65, 1, 128, dtype=torch.bfloat16, device="npu")
    inputs = [x.clone() for _ in range(4)] + [
        torch.ones(1, 65, 1, device="npu"),
        torch.zeros(1, device="npu"),
        torch.zeros(128, device="npu"),
    ]
    initial = torch.zeros(1, 1, 128, 128, device="npu")
    with pytest.raises(ValueError, match="must not alias"):
        _call(
            inputs,
            [0, 65],
            initial,
            offsets=torch.tensor([64], dtype=torch.int32, device="npu"),
            checkpoint=initial,
        )


@pytest.mark.npu
def test_unselected_boundaries_leave_all_rows_untouched():
    # None of these is a valid internal chunk64 boundary for a 65-token row.
    boundaries = [0, -64, 32, 65, 128]
    starts = [65 * i for i in range(len(boundaries) + 1)]
    x = torch.zeros(1, starts[-1], 2, 128, dtype=torch.bfloat16, device="npu")
    inputs = [x.clone() for _ in range(4)] + [
        torch.ones(1, starts[-1], 2, device="npu"),
        torch.zeros(2, device="npu"),
        torch.zeros(256, device="npu"),
    ]
    state = torch.zeros(len(boundaries), 2, 128, 128, device="npu")
    checkpoint = torch.full_like(state, 19.0)
    _call(
        inputs,
        starts,
        state,
        offsets=torch.tensor(boundaries, dtype=torch.int32, device="npu"),
        checkpoint=checkpoint,
    )
    torch.testing.assert_close(
        checkpoint, torch.full_like(checkpoint, 19.0), rtol=0, atol=0
    )


@pytest.mark.npu
@pytest.mark.parametrize("state_v_first", [True, False], ids=["VK", "KV"])
@pytest.mark.parametrize(
    "normalize", [True, False], ids=["fused-norm", "pre-normalized"]
)
def test_checkpoint_graph_replay_reads_updated_device_offsets(state_v_first, normalize):
    torch.npu.set_device(0)
    torch.manual_seed(29)
    inputs = [
        torch.randn(1, 193, 2, 128, dtype=torch.bfloat16, device="npu")
        for _ in range(4)
    ]
    if not normalize:
        inputs[:2] = [torch.nn.functional.normalize(t, dim=-1) for t in inputs[:2]]
    inputs += [
        torch.rand(1, 193, 2, device="npu"),
        torch.zeros(2, device="npu"),
        torch.zeros(256, device="npu"),
    ]
    initial = torch.randn(1, 2, 128, 128, device="npu")
    checkpoint = torch.full_like(initial, 19.0)
    offsets = torch.tensor([64], dtype=torch.int32, device="npu")

    def execute(destination):
        return _call(
            inputs,
            [0, 193],
            initial,
            offsets=offsets,
            checkpoint=destination,
            state_v_first=state_v_first,
            use_qk_l2norm_in_kernel=normalize,
        )

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            execute(checkpoint)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = execute(checkpoint)
    for offset in (64, 128, 0):
        with torch.npu.stream(stream):
            offsets.fill_(offset)
            inputs[2].mul_(0.9)
            expected_checkpoint = torch.full_like(checkpoint, 19.0)
            expected = execute(expected_checkpoint)
            checkpoint.fill_(19.0)
            graph.replay()
        torch.npu.synchronize()
        for index in (0, 1):
            torch.testing.assert_close(captured[index], expected[index], rtol=0, atol=0)
        torch.testing.assert_close(checkpoint, expected_checkpoint, rtol=0, atol=0)
    graph.reset()


@pytest.mark.npu
@pytest.mark.parametrize("state_v_first", [True, False], ids=["VK", "KV"])
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "absent"])
@pytest.mark.parametrize("offset", [64, 128])
def test_checkpoint_against_independent_fp64_fwd_h(state_v_first, warm, offset):
    torch.npu.set_device(0)
    torch.manual_seed(37)
    inputs = [
        torch.randn(1, 193, 2, 128, dtype=torch.bfloat16, device="npu")
        for _ in range(4)
    ] + [
        torch.rand(1, 193, 2, device="npu"),
        torch.full((2,), -4.0, device="npu"),
        torch.zeros(256, device="npu"),
    ]
    initial = torch.randn(1, 2, 128, 128, device="npu") * 0.02 if warm else None
    checkpoint = torch.full((1, 2, 128, 128), 19.0, device="npu")
    got = _call(
        inputs,
        [0, 193],
        initial,
        offsets=torch.tensor([offset], dtype=torch.int32, device="npu"),
        checkpoint=checkpoint,
        state_v_first=state_v_first,
        save_prepared=True,
    )
    expected = _fwd_h_fp64_reference(
        got[8], got[5], got[6], got[2], initial, [0, 193], [offset], state_v_first
    )
    for actual, reference in zip((got[1], checkpoint), expected):
        assert reference.dtype == torch.float64
        matched = torch.isclose(
            actual.cpu().double(), reference, rtol=9.77e-4, atol=1.53e-5
        )
        assert matched.double().mean().item() >= 0.99


@pytest.mark.npu
def test_final_boundary_does_not_write_internal_checkpoint():
    x = torch.zeros(1, 64, 2, 128, dtype=torch.bfloat16, device="npu")
    inputs = [x.clone() for _ in range(4)] + [
        torch.ones(1, 64, 2, device="npu"),
        torch.zeros(2, device="npu"),
        torch.zeros(256, device="npu"),
    ]
    checkpoint = torch.full((1, 2, 128, 128), 19.0, device="npu")
    _call(
        inputs,
        [0, 64],
        None,
        offsets=torch.tensor([64], dtype=torch.int32, device="npu"),
        checkpoint=checkpoint,
    )
    torch.testing.assert_close(
        checkpoint, torch.full_like(checkpoint, 19.0), rtol=0, atol=0
    )


@pytest.mark.npu
def test_checkpoint_export_without_final_state():
    torch.manual_seed(41)
    inputs = [
        torch.randn(1, 129, 2, 128, dtype=torch.bfloat16, device="npu")
        for _ in range(4)
    ] + [
        torch.rand(1, 129, 2, device="npu"),
        torch.full((2,), -4.0, device="npu"),
        torch.zeros(256, device="npu"),
    ]
    checkpoint = torch.full((1, 2, 128, 128), 19.0, device="npu")
    got = _call(
        inputs,
        [0, 129],
        None,
        offsets=torch.tensor([64], dtype=torch.int32, device="npu"),
        checkpoint=checkpoint,
        save_prepared=True,
        output_final_state=False,
    )
    assert got[1] is None
    _, expected = _fwd_h_fp64_reference(
        got[8], got[5], got[6], got[2], None, [0, 129], [64], True
    )
    matched = torch.isclose(
        checkpoint.cpu().double(), expected, rtol=9.77e-4, atol=1.53e-5
    )
    assert matched.double().mean().item() >= 0.99
