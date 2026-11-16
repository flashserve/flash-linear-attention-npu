"""Shape-only placeholder; this is NOT a numerical golden reference."""
import torch


def kda_gate_bwd_post_cpu(dg_act, raw_g, a_log, dt_bias, cu_seqlens,
                          chunk_indices, chunk_size, safe_gate, lower_bound,
                          input_scanned):
    heads = int(raw_g.shape[-3])
    key_dim = int(raw_g.shape[-1])
    # Output order follows the C++ OpDef: dg, dA, dbias.
    return (torch.zeros(tuple(raw_g.shape), dtype=torch.float32),
            torch.zeros((heads,), dtype=torch.float32),
            torch.zeros((heads * key_dim,), dtype=torch.float32))
