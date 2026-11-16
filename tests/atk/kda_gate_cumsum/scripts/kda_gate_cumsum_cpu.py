"""CPU reference for the standalone head-major gate-cumsum contract.

The FP64 calculation follows ApplyGate and ProcessChunk in the C++ kernel.
This reference has not undergone NPU precision/value-range calibration.
"""
import math
import torch


def kda_gate_cumsum_cpu(g, chunk_size, a_log=None, dt_bias=None,
                        cu_seqlens=None, use_gate_in_kernel=False,
                        safe_gate=False, lower_bound=-5.0):
    rank3 = g.ndim == 3
    values = g.detach().cpu().to(torch.float64)
    if rank3:
        values = values.unsqueeze(0)
    batch, heads, tokens, key_dim = values.shape
    if use_gate_in_kernel:
        exp_a = a_log.detach().cpu().to(torch.float64).exp().reshape(1, heads, 1, 1)
        if dt_bias is not None:
            values = values + dt_bias.detach().cpu().to(torch.float64).reshape(1, heads, 1, key_dim)
        if safe_gate:
            values = float(lower_bound) * torch.sigmoid(exp_a * values)
        else:
            # Stable softplus without the default threshold approximation.
            values = -exp_a * torch.logaddexp(values, torch.zeros_like(values))
    values = values / math.log(2.0)
    output = torch.empty_like(values)
    if cu_seqlens is None:
        sequences = [(b, 0, tokens) for b in range(batch)]
    else:
        offsets = [int(x) for x in cu_seqlens]
        sequences = [(0, start, end) for start, end in zip(offsets, offsets[1:])]
    for batch_index, start, end in sequences:
        for begin in range(start, end, int(chunk_size)):
            stop = min(begin + int(chunk_size), end)
            output[batch_index, :, begin:stop] = values[batch_index, :, begin:stop].cumsum(dim=1)
    return (output.squeeze(0) if rank3 else output).to(torch.float32)
