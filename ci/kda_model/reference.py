"""Independent FP32 KDA reference. All inputs and outputs are token-major on CPU."""
import torch


def block(inputs, state, scale, bound):
    # One short block per head. Solve the causal delta rule for all value channels.
    q, k, v, raw, beta, a, bias = inputs
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    cumulative = (bound * torch.sigmoid(a[:, None, None].exp() * (raw + bias[:, None]))).cumsum(1)
    decay = cumulative.exp()
    incoming_key = k * decay
    write_key = k * (-cumulative).exp()
    beta = beta.unsqueeze(-1)
    previous = (incoming_key @ write_key.transpose(1, 2)).tril(-1) * beta
    delta = torch.linalg.solve_triangular(
        previous, beta * (v - incoming_key @ state), upper=False, unitriangular=True)
    read = q * decay * scale
    output = read @ state + (read @ write_key.transpose(1, 2)).tril() @ delta
    final = decay[:, -1, :, None] * state + (
        k * (cumulative[:, -1:] - cumulative).exp()).transpose(1, 2) @ delta
    return output, final


def reference(tensors, grad_output, boundaries, bound):
    q, k, v, raw, beta, a, bias = (tensors[name] for name in ("q", "k", "v", "g", "beta", "A_log", "dt_bias"))
    batch, tokens, heads, width = q.shape
    flat = [x.flatten(0, 1) for x in (q, k, v, raw, beta)]
    dout = grad_output.flatten(0, 1)
    bounds = boundaries if boundaries is not None else range(0, (batch + 1) * tokens, tokens)
    outputs = torch.empty(dout.shape, dtype=torch.float32)
    gradients = [torch.empty(x.shape, dtype=torch.float32) for x in flat]
    finals = []
    # Checkpoint every 256 tokens. Rebuild only that window's autograd graph in reverse.
    # 8-token substeps bound exp(-cumulative) even at lower_bound=-5.
    def window(xs, state):
        pieces = []
        for pos in range(0, xs[0].size(1), 8):
            inputs = [x[:, pos:pos + 8] for x in xs] + [a, bias.reshape(heads, width)]
            out, state = block(inputs, state, width ** -0.5, bound)
            pieces.append(out)
        return torch.cat(pieces, dim=1), state

    for start, end in zip(bounds, bounds[1:]):
        state = torch.zeros(heads, width, v.size(-1))
        checkpoints = []
        for pos in range(start, end, 256):
            stop = min(pos + 256, end)
            xs = [x[pos:stop].transpose(0, 1).float() for x in flat]
            checkpoints.append((pos, stop, state))
            with torch.no_grad():
                out, state = window(xs, state)
                outputs[pos:stop] = out.transpose(0, 1)
        finals.append(state)
        carry = torch.zeros_like(state)  # no terminal-state gradient in this model loss
        for pos, stop, state in reversed(checkpoints):
            xs = [x[pos:stop].transpose(0, 1).float().detach().requires_grad_() for x in flat]
            state = state.detach().requires_grad_()
            out, next_state = window(xs, state)
            *grads, carry = torch.autograd.grad(
                (out, next_state), (*xs, state), (dout[pos:stop].transpose(0, 1).float(), carry))
            for dst, grad in zip(gradients, grads):
                dst[pos:stop] = grad.transpose(0, 1)
    dq, dk, dv, dg, dbeta = [x.reshape(batch, tokens, *x.shape[1:]) for x in gradients]
    return dict(o=outputs.reshape_as(v), final_state=torch.stack(finals), dq=dq, dk=dk,
                dv=dv, dg=dg, dbeta=dbeta)
