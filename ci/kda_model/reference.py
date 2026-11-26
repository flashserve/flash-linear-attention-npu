"""CPU FP32 golden copied from examples/chunk_kda.py::recurrent_reference (79164d12).

The token recurrence and internal [H,V,K] layout follow the original. The CI
adapter carries the state across memory checkpoints and exports it as [N,H,K,V].
"""
import torch


def recurrent_window(q, k, v, raw_gate, beta, a_log, dt_bias, state, *, scale, lower_bound,
                     qk_l2norm=True):
    """Run the original recurrence on one window of one sequence (batch size 1)."""
    q_raw = q.float()
    k_raw = k.float()
    v = v.float()
    gate_input = raw_gate.float() + dt_bias.float().reshape(a_log.numel(), -1).unsqueeze(0).unsqueeze(0)
    decay = torch.exp(a_log.float()).view(1, 1, -1, 1)

    leaves = {
        "q": q_raw.clone().requires_grad_(True),
        "k": k_raw.clone().requires_grad_(True),
        "v": v.clone().requires_grad_(True),
        "raw_gate": gate_input.detach().clone().requires_grad_(True),
        "beta": beta.float().clone().requires_grad_(True),
    }
    # 参考内部做归一化（与 kernel 的 use_qk_l2norm_in_kernel 口径一致），
    # 这样 autograd 给出的 dq/dk 也是对**原始** q/k 的梯度，可与算子输出直接对比。
    if qk_l2norm:
        q_norm = leaves["q"] * torch.rsqrt(
            leaves["q"].square().sum(-1, keepdim=True) + 1e-6)
        k_norm = leaves["k"] * torch.rsqrt(
            leaves["k"].square().sum(-1, keepdim=True) + 1e-6)
    else:
        q_norm, k_norm = leaves["q"], leaves["k"]
    # gate 必须由叶子张量算出，否则 raw_gate 不在反传图上。
    gate = lower_bound * torch.sigmoid(decay * leaves["raw_gate"])
    outputs = []
    batch_id = 0
    for token_id in range(q.size(1)):
        # decay 按 key 维广播：state [H,V,K] × gate [H,K] → unsqueeze(1)。
        state = state * torch.exp(gate[batch_id, token_id]).unsqueeze(1)
        key = k_norm[batch_id, token_id]
        delta = (leaves["v"][batch_id, token_id]
                 - torch.einsum("hvk,hk->hv", state, key)) \
            * leaves["beta"][batch_id, token_id].unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * key.unsqueeze(1)
        outputs.append(torch.einsum(
            "hvk,hk->hv", state, q_norm[batch_id, token_id] * scale))
    return torch.stack(outputs).unsqueeze(0), state, leaves


def reference(tensors, grad_output, boundaries, bound):
    q, k, v, raw, beta = [tensors[name] for name in ("q", "k", "v", "g", "beta")]
    batch, tokens, heads, kd = q.shape
    vd = v.size(-1)
    flat = [x.flatten(0, 1) for x in (q, k, v, raw, beta)]
    dout = grad_output.flatten(0, 1)
    bounds = boundaries if boundaries is not None else range(0, (batch + 1) * tokens, tokens)
    outputs = torch.empty(dout.shape, dtype=torch.float32)
    gradients = [torch.empty(x.shape, dtype=torch.float32) for x in flat]
    finals = []

    def window(pos, stop, state):
        xs = [x[pos:stop].unsqueeze(0) for x in flat]
        return recurrent_window(*xs, tensors["A_log"], tensors["dt_bias"], state,
                                scale=kd ** -0.5, lower_bound=bound)

    # Preserve the recurrent state within a sequence, including across checkpoints.
    # Dense batch elements and separate packed sequences each start at zero.
    for start, end in zip(bounds, bounds[1:]):
        state = torch.zeros(heads, vd, kd, dtype=torch.float32)
        checkpoints = []
        with torch.no_grad():
            for pos in range(start, end, 256):
                stop = min(pos + 256, end)
                checkpoints.append((pos, stop, state))
                out, state, _ = window(pos, stop, state)
                outputs[pos:stop] = out[0]
        finals.append(state.transpose(-1, -2))
        carry = torch.zeros_like(state)
        for pos, stop, state in reversed(checkpoints):
            state = state.detach().requires_grad_()
            out, next_state, leaves = window(pos, stop, state)
            inputs = [leaves[name] for name in ("q", "k", "v", "raw_gate", "beta")]
            *grads, carry = torch.autograd.grad(
                (out, next_state), (*inputs, state), (dout[pos:stop].float().unsqueeze(0), carry))
            for dst, grad in zip(gradients, grads):
                dst[pos:stop] = grad[0]
    dq, dk, dv, dg, dbeta = [x.reshape(batch, tokens, *x.shape[1:]) for x in gradients]
    return dict(o=outputs.reshape_as(v), final_state=torch.stack(finals), dq=dq, dk=dk,
                dv=dv, dg=dg, dbeta=dbeta)
