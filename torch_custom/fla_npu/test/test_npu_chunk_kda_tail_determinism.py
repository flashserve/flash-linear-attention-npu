import pytest
import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import chunk_kda_fwd


CHUNK_SIZE = 64
DETERMINISM_REPEATS = 20
OUTPUT_NAMES = (
    "attn_out",
    "final_state",
    "gk",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state_out",
)


def _chunk_indices(tokens):
    return [
        value
        for chunk_id in range((tokens + CHUNK_SIZE - 1) // CHUNK_SIZE)
        for value in (0, chunk_id)
    ]


def _snapshot(outputs):
    torch.npu.synchronize()
    return tuple(
        None if output is None else output.detach().cpu().contiguous()
        for output in outputs
    )


def _assert_bitwise_equal(expected, actual, repeat):
    assert len(expected) == len(actual) == len(OUTPUT_NAMES)
    for name, expected_output, actual_output in zip(OUTPUT_NAMES, expected, actual):
        if expected_output is None or actual_output is None:
            assert expected_output is None and actual_output is None, (
                f"repeat={repeat} output={name} changed between None and Tensor"
            )
            continue
        same_metadata = (
            expected_output.shape == actual_output.shape
            and expected_output.dtype == actual_output.dtype
        )
        same_bits = same_metadata and torch.equal(
            expected_output.view(torch.uint8), actual_output.view(torch.uint8)
        )
        assert same_bits, f"repeat={repeat} output={name} is not bitwise deterministic"


@pytest.mark.parametrize(
    ("tokens", "disable_recompute"),
    [
        pytest.param(15, False, id="single-tail-model-mode"),
        pytest.param(15, True, id="single-tail-all-outputs"),
        pytest.param(65, True, id="full-chunk-plus-tail-all-outputs"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_tail_is_bitwise_deterministic(tokens, disable_recompute):
    torch.manual_seed(20260820 + tokens + int(disable_recompute))
    torch.npu.set_device(0)

    shape = (1, tokens, 6, 128)
    q = (torch.randn(shape) * 0.04).to(torch.bfloat16).npu()
    k = (torch.randn(shape) * 0.04).to(torch.bfloat16).npu()
    v = (torch.randn(shape) * 0.04).to(torch.bfloat16).npu()
    raw_gate = (-7.0 + torch.randn(shape) * 0.03).to(torch.float32).npu()
    beta = (torch.rand((1, tokens, 6)) * 0.2 + 0.05).to(torch.float32).npu()
    initial_state = (torch.randn((1, 6, 128, 128)) * 0.01).to(torch.float32).npu()
    a_log = torch.zeros(6, dtype=torch.float32, device="npu")
    dt_bias = torch.zeros(6 * 128, dtype=torch.float32, device="npu")

    def run():
        return chunk_kda_fwd(
            q,
            k,
            v,
            raw_gate,
            beta,
            128**-0.5,
            CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=[0, tokens],
            chunk_indices=_chunk_indices(tokens),
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            disable_recompute=disable_recompute,
            return_intermediate_states=True,
            state_v_first=True,
        )

    run()
    expected = _snapshot(run())
    for repeat in range(1, DETERMINISM_REPEATS):
        _assert_bitwise_equal(expected, _snapshot(run()), repeat)
