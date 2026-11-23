from __future__ import annotations

import math
from itertools import accumulate

import torch


def validate_mega_recurrent_kda_inputs(
    hidden_states,
    qkv_projection_weight,
    decay_projection_a_weight,
    decay_projection_b_weight,
    beta_projection_weight,
    output_gate_projection_weight,
    output_norm_weight,
    output_projection_weight,
    conv1d_weight,
    conv_state,
    checkpoint_state,
    replay_u,
    replay_k,
    replay_decay,
    state_indices,
    conv_num_accepted_tokens,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    rms_norm_eps,
    cu_seqlens,
):
    """Validate the complete logical ND input contract before computation."""
    named_tensors = {
        "hidden_states": hidden_states,
        "qkv_projection_weight": qkv_projection_weight,
        "decay_projection_a_weight": decay_projection_a_weight,
        "decay_projection_b_weight": decay_projection_b_weight,
        "beta_projection_weight": beta_projection_weight,
        "output_gate_projection_weight": output_gate_projection_weight,
        "output_norm_weight": output_norm_weight,
        "output_projection_weight": output_projection_weight,
        "conv1d_weight": conv1d_weight,
        "conv_state": conv_state,
        "checkpoint_state": checkpoint_state,
        "replay_u": replay_u,
        "replay_k": replay_k,
        "replay_decay": replay_decay,
        "state_indices": state_indices,
        "conv_num_accepted_tokens": conv_num_accepted_tokens,
        "A_log": A_log,
        "dt_bias": dt_bias,
    }
    if cu_seqlens is not None:
        named_tensors["cu_seqlens"] = cu_seqlens
    for name, value in named_tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor for this golden")

    if hidden_states.ndim not in (2, 3):
        raise ValueError("hidden_states must be [B,S,H] or [T,H]")
    if hidden_states.shape[-1] != 7168:
        raise ValueError("hidden_states last dimension must be H=7168")
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")

    batch = state_indices.numel()
    if batch <= 0:
        raise ValueError("batch must be positive")
    if hidden_states.ndim == 3:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens must be None for BSH input")
        if hidden_states.shape[0] != batch:
            raise ValueError("hidden_states B must equal state_indices length")
        if hidden_states.shape[1] <= 0:
            raise ValueError("sequence length must be positive")
        prefix = [batch_index * hidden_states.shape[1] for batch_index in range(batch + 1)]
        total_tokens = batch * hidden_states.shape[1]
    else:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for TH input")
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.shape[0] != batch + 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError("cu_seqlens must be contiguous int32/int64 [B+1]")
        prefix = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
        total_tokens = hidden_states.shape[0]
        if total_tokens <= 0:
            raise ValueError("TH token count must be positive")

    if prefix[0] != 0 or prefix[-1] != total_tokens:
        raise ValueError("sequence prefix must run from 0 to total token count")
    if not all(left < right for left, right in zip(prefix, prefix[1:])):
        raise ValueError("every sequence must be non-empty")
    lengths = [right - left for left, right in zip(prefix, prefix[1:])]

    if beta_projection_weight.ndim != 2:
        raise ValueError("beta_projection_weight must be [N,H]")
    heads = beta_projection_weight.shape[0]
    dim = 128
    if heads < 1:
        raise ValueError("N must be >= 1")
    projection_width = heads * dim

    expected_weights = (
        (qkv_projection_weight, (3 * projection_width, 7168), torch.bfloat16),
        (decay_projection_a_weight, (dim, 7168), torch.bfloat16),
        (decay_projection_b_weight, (projection_width, dim), torch.bfloat16),
        (beta_projection_weight, (heads, 7168), torch.bfloat16),
        (
            output_gate_projection_weight,
            (projection_width, 7168),
            torch.bfloat16,
        ),
        (output_norm_weight, (dim,), torch.float32),
        (output_projection_weight, (7168, projection_width), torch.bfloat16),
    )
    for tensor, shape, dtype in expected_weights:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"projection tensor must have shape {shape} and dtype {dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError("projection tensors must be contiguous")

    if tuple(conv1d_weight.shape) not in (
        (4, 3 * projection_width),
        (3, heads, 4, dim),
        (3, 4, dim),
    ):
        raise ValueError("invalid conv1d_weight shape")
    if conv1d_weight.dtype != torch.bfloat16 or not conv1d_weight.is_contiguous():
        raise ValueError("conv1d_weight must be contiguous BF16")

    if conv_state.ndim != 3 or conv_state.shape[2] != 3 * projection_width:
        raise ValueError("conv_state must be [block_num,L,3*P]")
    if conv_state.dtype != torch.bfloat16:
        raise ValueError("conv_state must be BF16")
    block_num = conv_state.shape[0]
    conv_length = conv_state.shape[1]
    if block_num <= 0 or conv_length < 3:
        raise ValueError("conv_state requires positive block_num and L>=3")
    if max(lengths) + 2 > conv_length:
        raise ValueError("conv_state L must cover max sequence length + 2")

    if (
        tuple(checkpoint_state.shape) != (block_num, heads, dim, dim)
        or checkpoint_state.dtype != torch.float32
    ):
        raise ValueError("checkpoint_state must be FP32 [block_num,N,D,D]")
    if replay_u.ndim != 4:
        raise ValueError("replay_u must be [block_num,N,S,D]")
    replay_shape = (block_num, heads, replay_u.shape[2], dim)
    for name, tensor, dtype in (
        ("replay_u", replay_u, torch.float32),
        ("replay_k", replay_k, torch.float16),
        ("replay_decay", replay_decay, torch.float32),
    ):
        if tuple(tensor.shape) != replay_shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype} [block_num,N,S,D]")
    replay_capacity = replay_shape[2]
    if replay_capacity <= 0 or max(lengths) > replay_capacity:
        raise ValueError("replay S must cover the longest sequence")

    for name, tensor in (
        ("conv_state", conv_state),
        ("checkpoint_state", checkpoint_state),
        ("replay_u", replay_u),
        ("replay_k", replay_k),
        ("replay_decay", replay_decay),
    ):
        expected_stride = 1
        for axis in range(tensor.ndim - 1, 0, -1):
            if tensor.stride(axis) != expected_stride:
                raise ValueError(f"{name} only axis 0 may be non-contiguous")
            expected_stride *= tensor.shape[axis]
        if tensor.stride(0) < expected_stride:
            raise ValueError(f"{name} block storage must not overlap")

    if (
        state_indices.ndim != 1
        or tuple(state_indices.shape) != (batch,)
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
    ):
        raise ValueError("state_indices must be contiguous int32 [B]")
    indices = [int(value) for value in state_indices.detach().cpu().tolist()]
    if any(value < 0 or value >= block_num for value in indices):
        raise ValueError("state_indices values must be in [0,block_num)")
    if len(set(indices)) != batch:
        raise ValueError("state_indices must be distinct")

    if (
        conv_num_accepted_tokens.ndim != 1
        or tuple(conv_num_accepted_tokens.shape) != (batch,)
        or conv_num_accepted_tokens.dtype != torch.int32
        or not conv_num_accepted_tokens.is_contiguous()
    ):
        raise ValueError("conv_num_accepted_tokens must be contiguous int32 [B]")
    accepted = [
        int(value)
        for value in conv_num_accepted_tokens.detach().cpu().tolist()
    ]
    if any(value < 0 or value > conv_length - 2 for value in accepted):
        raise ValueError("conv_num_accepted_tokens must be in [0,L-2]")

    if tuple(A_log.shape) != (heads,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [N]")
    if tuple(dt_bias.shape) != (heads, dim) or dt_bias.dtype != torch.float32:
        raise ValueError("dt_bias must be FP32 [N,D]")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("A_log and dt_bias must be contiguous")

    for name, value in (
        ("scale", scale),
        ("lower_bound", lower_bound),
        ("rms_norm_eps", rms_norm_eps),
    ):
        narrowed = float(torch.tensor(float(value), dtype=torch.float32).item())
        if not math.isfinite(float(value)) or not math.isfinite(narrowed):
            raise ValueError(f"{name} must be finite and representable as FP32")
    if not -5.0 <= float(lower_bound) <= 0.0:
        raise ValueError("lower_bound must be in [-5,0]")
    if float(rms_norm_eps) < float.fromhex("0x1.0p-126"):
        raise ValueError("rms_norm_eps must be a normal positive FP32 value")

    for name, tensor in (
        ("conv_state", conv_state),
        ("replay_u", replay_u),
        ("replay_k", replay_k),
        ("replay_decay", replay_decay),
    ):
        if tensor.requires_grad:
            raise ValueError(f"{name} must be an inference-only writable buffer")


def mega_recurrent_kda_golden(
    hidden_states,
    qkv_projection_weight,
    decay_projection_a_weight,
    decay_projection_b_weight,
    beta_projection_weight,
    output_gate_projection_weight,
    output_norm_weight,
    output_projection_weight,
    conv1d_weight,
    conv_state,
    checkpoint_state,
    replay_u,
    replay_k,
    replay_decay,
    state_indices,
    conv_num_accepted_tokens,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    rms_norm_eps=1e-5,
    cu_seqlens=None,
):
    """Run the complete Mega KDA forward in FP32 and update the supplied pools."""
    validate_mega_recurrent_kda_inputs(
        hidden_states,
        qkv_projection_weight,
        decay_projection_a_weight,
        decay_projection_b_weight,
        beta_projection_weight,
        output_gate_projection_weight,
        output_norm_weight,
        output_projection_weight,
        conv1d_weight,
        conv_state,
        checkpoint_state,
        replay_u,
        replay_k,
        replay_decay,
        state_indices,
        conv_num_accepted_tokens,
        A_log,
        dt_bias,
        scale,
        lower_bound,
        rms_norm_eps,
        cu_seqlens,
    )

    hidden_width = 7168
    dim = 128
    heads = beta_projection_weight.shape[0]
    projection_width = heads * dim
    batch = state_indices.numel()
    if hidden_states.ndim == 3:
        prefix = [batch_index * hidden_states.shape[1] for batch_index in range(batch + 1)]
    else:
        prefix = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    indices = [int(value) for value in state_indices.detach().cpu().tolist()]
    accepted = [
        int(value)
        for value in conv_num_accepted_tokens.detach().cpu().tolist()
    ]

    x_all = hidden_states.reshape(-1, hidden_width).detach().to(torch.float32)
    qkv_raw = x_all @ qkv_projection_weight.detach().to(torch.float32).T
    decay_a = (
        x_all @ decay_projection_a_weight.detach().to(torch.float32).T
    ).to(torch.bfloat16)
    decay_raw = (
        decay_a.to(torch.float32)
        @ decay_projection_b_weight.detach().to(torch.float32).T
    ).to(torch.bfloat16)
    beta_raw = (
        x_all @ beta_projection_weight.detach().to(torch.float32).T
    ).to(torch.bfloat16)
    gate_raw = (
        x_all @ output_gate_projection_weight.detach().to(torch.float32).T
    ).to(torch.bfloat16)

    if tuple(conv1d_weight.shape) == (4, 3 * projection_width):
        conv_weight = (
            conv1d_weight.detach()
            .to(torch.float32)
            .reshape(4, 3, heads, dim)
            .permute(1, 2, 0, 3)
        )
    elif tuple(conv1d_weight.shape) == (3, heads, 4, dim):
        conv_weight = conv1d_weight.detach().to(torch.float32)
    else:
        conv_weight = (
            conv1d_weight.detach()
            .to(torch.float32)[:, None]
            .expand(3, heads, 4, dim)
        )

    out_tokens = torch.empty(
        (x_all.shape[0], hidden_width), dtype=torch.float32, device=x_all.device
    )
    gamma = output_norm_weight.detach().to(torch.float32)
    output_projection = output_projection_weight.detach().to(torch.float32)
    alpha = torch.exp(A_log.detach().to(torch.float32))
    bias = dt_bias.detach().to(torch.float32)

    for batch_index in range(batch):
        start = prefix[batch_index]
        end = prefix[batch_index + 1]
        sequence = end - start
        slot = indices[batch_index]

        history_start = max(accepted[batch_index] - 1, 0)
        history = conv_state[
            slot, history_start : history_start + 3
        ].clone().detach().to(torch.float32)
        raw_i = qkv_raw[start:end].reshape(sequence, 3, heads, dim)
        raw_history = raw_i.to(torch.bfloat16).to(torch.float32)
        sliding = torch.cat(
            (history.reshape(3, 3, heads, dim), raw_history), dim=0
        )

        convolved = torch.empty_like(raw_i)
        for token in range(sequence):
            value = sliding.new_zeros((3, heads, dim))
            for tap in range(4):
                value = value + sliding[token + tap] * conv_weight[:, :, tap, :]
            convolved[token] = torch.nn.functional.silu(value).to(torch.bfloat16)

        conv_state[slot, 0:2].copy_(
            history[1:3].to(conv_state.dtype)
        )
        conv_state[slot, 2 : 2 + sequence].copy_(
            raw_i.reshape(sequence, 3 * projection_width).to(conv_state.dtype)
        )

        query = convolved[:, 0].to(torch.float32)
        key = convolved[:, 1].to(torch.float32)
        value = convolved[:, 2].to(torch.float32)
        query = query * torch.rsqrt(
            query.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        key = key * torch.rsqrt(
            key.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        query = query * float(scale)
        beta = torch.sigmoid(beta_raw[start:end].to(torch.float32))
        raw_gate = decay_raw[start:end].to(torch.float32).reshape(
            sequence, heads, dim
        )
        gate_logits = alpha.reshape(1, heads, 1) * (
            raw_gate + bias.reshape(1, heads, dim)
        )
        decay = torch.exp(float(lower_bound) * torch.sigmoid(gate_logits))
        output_gate = gate_raw[start:end].to(torch.float32).reshape(
            sequence, heads, dim
        )

        recurrent_output = torch.empty(
            (sequence, heads, dim), dtype=torch.float32, device=x_all.device
        )
        for head in range(heads):
            state = checkpoint_state[slot, head].detach().to(torch.float32)
            for token in range(sequence):
                decay_t = decay[token, head]
                key_t = key[token, head]
                query_t = query[token, head]
                value_t = value[token, head]
                decayed_state = state * decay_t.reshape(1, dim)
                correction = beta[token, head] * (
                    value_t - torch.mv(decayed_state, key_t)
                )
                state = decayed_state + torch.outer(correction, key_t)
                recurrent_output[token, head] = torch.mv(state, query_t)

                replay_u[slot, head, token].copy_(correction.to(replay_u.dtype))
                replay_k[slot, head, token].copy_(key_t.to(replay_k.dtype))
                replay_decay[slot, head, token].copy_(decay_t.to(replay_decay.dtype))

        gated = torch.empty_like(recurrent_output)
        for token in range(sequence):
            inverse_rms = torch.rsqrt(
                recurrent_output[token].square().mean(dim=-1, keepdim=True)
                + float(rms_norm_eps)
            )
            normalized = recurrent_output[token] * inverse_rms * gamma.reshape(1, -1)
            gated[token] = (
                normalized * torch.sigmoid(output_gate[token])
            ).to(torch.bfloat16)

        projected = (
            gated.reshape(sequence, projection_width) @ output_projection.T
        )
        out_tokens[start:end] = projected

    return (
        out_tokens.reshape(hidden_states.shape).to(hidden_states.dtype),
        conv_state,
        replay_u,
        replay_k,
        replay_decay,
    )


def _make_case(batch, sequence, heads, block_num=4, conv_weight_shape="flat", accepted=None, seed=0):
    """Build one deterministic uniform-length case plus its BSH/TH input views."""
    torch.manual_seed(seed)
    dim, hidden = 128, 7168
    projection_width = heads * dim
    conv_length = sequence + 2
    total_tokens = batch * sequence
    if accepted is None:
        accepted = [0] * batch
    hidden_states = torch.randn(total_tokens, hidden).to(torch.bfloat16)
    qkv_weight = (torch.randn(3 * projection_width, hidden) * 0.02).to(torch.bfloat16)
    decay_a = (torch.randn(dim, hidden) * 0.02).to(torch.bfloat16)
    decay_b = (torch.randn(projection_width, dim) * 0.02).to(torch.bfloat16)
    beta_weight = (torch.randn(heads, hidden) * 0.02).to(torch.bfloat16)
    gate_weight = (torch.randn(projection_width, hidden) * 0.02).to(torch.bfloat16)
    norm_weight = torch.ones(dim)
    out_weight = (torch.randn(hidden, projection_width) * 0.02).to(torch.bfloat16)
    if conv_weight_shape == "flat":
        conv_weight = (torch.randn(4, 3 * projection_width) * 0.02).to(torch.bfloat16)
    elif conv_weight_shape == "per_head":
        conv_weight = (torch.randn(3, heads, 4, dim) * 0.02).to(torch.bfloat16)
    else:
        conv_weight = (torch.randn(3, 4, dim) * 0.02).to(torch.bfloat16)
    conv_state = torch.randn(block_num, conv_length, 3 * projection_width).to(torch.bfloat16)
    checkpoint = torch.randn(block_num, heads, dim, dim) * 0.1
    replay_u = torch.zeros(block_num, heads, sequence, dim)
    replay_k = torch.zeros(block_num, heads, sequence, dim, dtype=torch.float16)
    replay_decay = torch.zeros(block_num, heads, sequence, dim)
    base = dict(
        qkv_projection_weight=qkv_weight,
        decay_projection_a_weight=decay_a,
        decay_projection_b_weight=decay_b,
        beta_projection_weight=beta_weight,
        output_gate_projection_weight=gate_weight,
        output_norm_weight=norm_weight,
        output_projection_weight=out_weight,
        conv1d_weight=conv_weight,
        conv_state=conv_state,
        checkpoint_state=checkpoint,
        replay_u=replay_u,
        replay_k=replay_k,
        replay_decay=replay_decay,
        state_indices=torch.arange(batch, dtype=torch.int32),
        conv_num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros(heads, dim),
        scale=dim ** -0.5,
        lower_bound=-5.0,
    )
    cu_seqlens = torch.tensor(
        [index * sequence for index in range(batch + 1)], dtype=torch.int32
    )
    return base, hidden_states, cu_seqlens
