import math
from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import chunk_kda_fwd


CHUNK_SIZE = 64
HEADS = 6
HEAD_DIM = 128
LOWER_BOUND = -5.0
MAX_REASONABLE_ABS = 1.0e6
TAIL_TEST_TOKENS = (
    128,
    129,
    130,
    131,
    143,
    144,
    145,
    159,
    160,
    161,
    191,
    192,
    193,
)
OUTPUT_NAMES = (
    "output",
    "final_state",
    "gk",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state",
)


@dataclass
class ChunkKdaInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    activated_gate: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    initial_state: torch.Tensor
    cu_seqlens: tuple[int, ...]
    chunk_indices: tuple[int, ...]

    def clone(self):
        return ChunkKdaInputs(
            q=self.q.clone(),
            k=self.k.clone(),
            v=self.v.clone(),
            raw_gate=self.raw_gate.clone(),
            activated_gate=self.activated_gate.clone(),
            beta=self.beta.clone(),
            a_log=self.a_log.clone(),
            dt_bias=self.dt_bias.clone(),
            initial_state=self.initial_state.clone(),
            cu_seqlens=self.cu_seqlens,
            chunk_indices=self.chunk_indices,
        )


def l2norm(value):
    dtype = value.dtype
    value_fp32 = value.float()
    return (
        value_fp32
        * torch.rsqrt((value_fp32 * value_fp32).sum(dim=-1, keepdim=True) + 1e-6)
    ).to(dtype)


def build_inputs(tokens):
    seed = 20260820 + tokens
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

    shape = (1, tokens, HEADS, HEAD_DIM)
    q = l2norm(torch.randn(shape, device="npu", dtype=torch.bfloat16))
    k = l2norm(torch.randn(shape, device="npu", dtype=torch.bfloat16))
    v = torch.randn(shape, device="npu", dtype=torch.bfloat16) * 0.2
    raw_gate = torch.randn(shape, device="npu", dtype=torch.bfloat16) * 2.0
    beta = torch.sigmoid(
        torch.randn((1, tokens, HEADS), device="npu", dtype=torch.float32)
    )
    a_log = torch.empty((HEADS,), device="npu", dtype=torch.float32).uniform_(-0.5, 0.8)
    dt_bias = torch.empty(
        (HEADS * HEAD_DIM,), device="npu", dtype=torch.float32
    ).uniform_(-7.5, -1.5)
    initial_state = torch.zeros(
        (1, HEADS, HEAD_DIM, HEAD_DIM), device="npu", dtype=torch.float32
    )
    activated_gate = LOWER_BOUND * torch.sigmoid(
        (raw_gate.float() + dt_bias.view(1, 1, HEADS, HEAD_DIM))
        * a_log.exp().view(1, 1, HEADS, 1)
    )
    chunk_indices = tuple(
        value
        for chunk_index in range(math.ceil(tokens / CHUNK_SIZE))
        for value in (0, chunk_index)
    )
    return ChunkKdaInputs(
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        activated_gate=activated_gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=(0, tokens),
        chunk_indices=chunk_indices,
    )


def final_state_reference(inputs):
    k = inputs.k.detach().cpu()
    v = inputs.v.detach().cpu()
    gate = inputs.activated_gate.detach().cpu()
    beta = inputs.beta.detach().cpu()
    initial_state = inputs.initial_state.detach().cpu()
    _, tokens, heads, head_dim = k.shape
    final_state = torch.empty_like(initial_state, dtype=torch.float32)

    for head_index in range(heads):
        state_kv = initial_state[0, head_index].float().transpose(-1, -2).contiguous()
        for start in range(0, tokens, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, tokens)
            chunk_tokens = end - start
            strict_causal = torch.ones(
                (chunk_tokens, chunk_tokens), dtype=torch.bool
            ).tril(diagonal=-1)
            eye = torch.eye(chunk_tokens, dtype=torch.float32)
            k_block = k[0, start:end, head_index].float()
            v_block = v[0, start:end, head_index].float()
            beta_block = beta[0, start:end, head_index].float()
            gk_block = torch.cumsum(
                gate[0, start:end, head_index].float(), dim=0
            ) / math.log(2.0)
            relative_gate = gk_block[:, None, :] - gk_block[None, :, :]
            gate_factor = torch.exp2(
                relative_gate.masked_fill(~strict_causal[:, :, None], 0.0)
            )
            kk = torch.einsum("ik,jk,ijk->ij", k_block, k_block, gate_factor)
            strict_kk = torch.where(strict_causal, kk * beta_block[:, None], 0.0)
            akk_block = torch.linalg.solve_triangular(eye + strict_kk, eye, upper=False)
            w_block = akk_block @ (k_block * beta_block[:, None] * torch.exp2(gk_block))
            u_block = akk_block @ (v_block * beta_block[:, None])
            kg_block = k_block * torch.exp2(gk_block[-1][None, :] - gk_block)
            v_new_block = u_block - w_block @ state_kv
            state_kv = (
                torch.exp2(gk_block[-1])[:, None] * state_kv + kg_block.T @ v_new_block
            )
        final_state[0, head_index] = state_kv.transpose(-1, -2)
    return final_state


def run_chunk_kda(inputs, gate_mode, metadata_mode):
    use_gate_in_kernel = gate_mode == "raw_gate"
    gate = inputs.raw_gate if use_gate_in_kernel else inputs.activated_gate
    use_varlen_metadata = metadata_mode == "varlen"
    return chunk_kda_fwd(
        inputs.q,
        inputs.k,
        inputs.v,
        gate,
        inputs.beta,
        HEAD_DIM**-0.5,
        CHUNK_SIZE,
        layout="BSND",
        initial_state=inputs.initial_state,
        output_final_state=True,
        cu_seqlens=inputs.cu_seqlens if use_varlen_metadata else None,
        chunk_indices=inputs.chunk_indices if use_varlen_metadata else None,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=inputs.a_log if use_gate_in_kernel else None,
        dt_bias=inputs.dt_bias if use_gate_in_kernel else None,
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=True,
    )


def snapshot(outputs):
    torch.npu.synchronize()
    return tuple(
        output.detach().cpu().contiguous() if isinstance(output, torch.Tensor) else None
        for output in outputs
    )


def assert_bitwise_equal(expected, actual, tokens):
    assert len(expected) == len(actual) == len(OUTPUT_NAMES)
    for name, first, second in zip(OUTPUT_NAMES, expected, actual):
        if first is None or second is None:
            assert first is None and second is None, (
                f"tokens={tokens} output={name} changed between Tensor and None"
            )
            continue
        same_metadata = first.shape == second.shape and first.dtype == second.dtype
        same_bits = same_metadata and torch.equal(
            first.view(torch.uint8), second.view(torch.uint8)
        )
        assert same_bits, f"tokens={tokens} output={name} is not bitwise equal"
        first_fp32 = first.float()
        second_fp32 = second.float()
        assert torch.isfinite(first_fp32).all(), (
            f"tokens={tokens} output={name} contains non-finite values"
        )
        assert torch.isfinite(second_fp32).all(), (
            f"tokens={tokens} output={name} contains non-finite values"
        )
        assert first_fp32.abs().max().item() <= MAX_REASONABLE_ABS, (
            f"tokens={tokens} output={name} has an unreasonable magnitude"
        )
        assert second_fp32.abs().max().item() <= MAX_REASONABLE_ABS, (
            f"tokens={tokens} output={name} has an unreasonable magnitude"
        )


@pytest.mark.parametrize(
    "tokens",
    TAIL_TEST_TOKENS,
    ids=lambda tokens: f"tokens_{tokens}_remainder_{tokens % CHUNK_SIZE}",
)
@pytest.mark.parametrize("gate_mode", ["external_gate", "raw_gate"])
@pytest.mark.parametrize("metadata_mode", ["dense", "varlen"])
@torch.inference_mode()
def test_chunk_kda_tail_metadata_is_bitwise_deterministic(
    tokens, gate_mode, metadata_mode
):
    inputs = build_inputs(tokens)
    # Allocate both sets before the first launch so an out-of-bounds write from
    # that launch cannot change tensors allocated for the second invocation.
    first_inputs = inputs.clone()
    second_inputs = inputs.clone()
    first = snapshot(run_chunk_kda(first_inputs, gate_mode, metadata_mode))
    second = snapshot(run_chunk_kda(second_inputs, gate_mode, metadata_mode))
    torch.testing.assert_close(
        first[1],
        final_state_reference(inputs),
        rtol=3e-2,
        atol=3e-2,
        msg=(
            f"final_state accuracy failed for tokens={tokens}, "
            f"gate_mode={gate_mode}, metadata_mode={metadata_mode}"
        ),
    )
    assert_bitwise_equal(first, second, tokens)
