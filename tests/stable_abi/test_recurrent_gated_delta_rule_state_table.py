# Regression tests for ragged MTP state-table addressing.
#
# Ported from the one-NPU reproducer in vllm-project/vllm-ascend#17837 and the
# operator cases added by vllm-project/vllm-ascend#17839 (Co-authored-by:
# QwertyJack).  beta=0 and g=0 keep the recurrent state numerically unchanged,
# so every case is an exact indexing check (rtol=0): the second request must
# read and write its own row of the state table, never a neighbour's.
#
# Run on an NPU host:  pytest tests/stable_abi/test_recurrent_gated_delta_rule_state_table.py -q

import pytest
import torch

from fla_npu.ops.ascendc import npu_recurrent_gated_delta_rule

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires Ascend NPU")


def _make_case(lengths, accepted, width, state_dtype):
    num_requests = len(lengths)
    num_blocks = num_requests * width + 2
    table = torch.arange(1, num_blocks - 1, dtype=torch.int32).view(-1, width)
    state = torch.arange(num_blocks, dtype=state_dtype)
    state = state[:, None, None, None].expand(-1, 4, 128, 128).contiguous()

    total = sum(lengths)
    query = torch.zeros(total, 1, 128, dtype=torch.bfloat16)
    query[..., 0] = 1
    value = torch.zeros(total, 4, 128, dtype=torch.bfloat16)
    beta = torch.zeros(total, 4, dtype=torch.bfloat16)
    g = torch.zeros(total, 4, dtype=torch.float32)

    # beta=0, g=0: the state passes through unchanged, so the expected output
    # of every token is the initial state block picked by accepted-1, and the
    # expected writeback fills slots 0..length-1 of each request row.
    expected_output = torch.empty_like(value)
    expected_state = state.clone()
    start = 0
    for row, length in enumerate(lengths):
        if length == 0:
            continue
        source = state[table[row, accepted[row] - 1]]
        expected_output[start:start + length] = source[..., 0]
        expected_state[table[row, :length]] = source
        start += length
    return table, state, query, value, beta, g, expected_output, expected_state


def _run_case(lengths, accepted, width, state_dtype, table_mode):
    table, state, query, value, beta, g, expected_output, expected_state = (
        _make_case(lengths, accepted, width, state_dtype))
    state_npu = state.npu()
    indices = table.npu() if table_mode == "table" else table.flatten().npu()
    output = npu_recurrent_gated_delta_rule(
        query.npu(),
        key=torch.zeros_like(query).npu(),
        value=value.npu(),
        state=state_npu,
        beta=beta.npu(),
        scale=1.0,
        actual_seq_lengths=torch.tensor([0] + lengths, dtype=torch.int32).npu(),
        ssm_state_indices=indices,
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32).npu(),
        g=g.npu(),
    )
    torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state_npu.cpu(), expected_state, rtol=0, atol=0)


CASES = [
    ([4, 4], [1, 1], 4),      # equal-length control
    ([2, 4], [1, 1], 4),      # the deterministic cross-request counterexample
    ([2, 3], [1, 1], 3),
    ([2, 2], [1, 1], 2),
    ([2, 4], [4, 3], 4),      # previous acceptance > current query length
    ([1, 4], [4, 1], 4),
    ([0, 2, 4], [0, 4, 2], 4),  # zero-length padding row
]


@pytest.mark.parametrize("lengths,accepted,width", CASES)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_state_table_two_dim(lengths, accepted, width, state_dtype):
    """[B, W] tables must preserve request ownership of state slots."""
    _run_case(lengths, accepted, width, state_dtype, "table")


@pytest.mark.parametrize("lengths,accepted,width", CASES)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_flat_indices_legacy(lengths, accepted, width, state_dtype):
    """The legacy per-token [T] contract keeps the vLLM 0.26 behaviour.

    Only equal-length batches are exact under the flat contract; ragged flat
    batches are excluded from the exact assertion because per-token addressing
    is their defined (legacy) behaviour.
    """
    if len(set(lengths)) > 1 or 0 in lengths:
        pytest.skip("flat contract is only exactly defined for equal lengths")
    _run_case(lengths, accepted, width, state_dtype, "flat")
