"""Operator-local input compatibility and metadata readiness for recurrent_kda.

The operator itself selects packed mode exclusively by cu_seqlens presence.
Keep these aliases through v26.12; remove no earlier than v27.1.
"""

import warnings


def wait_for_metadata(q, cu_seqlens, ssm_state_indices):
    """Complete framework producers before ACLNN reads device metadata on host.

    ACL runtime synchronization alone cannot flush the framework's pending
    submissions. Other streams must establish their dependency on this stream,
    as for ordinary tensor inputs. Dense calls without metadata stay asynchronous.
    """
    if (cu_seqlens is not None or ssm_state_indices is not None) and q.device.type == "npu":
        import torch

        torch.npu.current_stream(q.device).synchronize()


def normalize_inputs(q, k, v, g, beta, cu_seqlens):
    rank = q.ndim
    legacy = (rank == 3 and cu_seqlens is None) or (rank == 4 and cu_seqlens is not None)
    if not legacy:
        return q, k, v, g, beta, None
    prefix = 1 if rank == 3 else 2
    tensors = (q, k, v, g, beta)
    if (any(t.ndim != rank for t in (k, v, g)) or beta.ndim != rank - 1
            or q.shape != k.shape
            or any(t.shape[:prefix] != q.shape[:prefix] for t in tensors[2:])):
        raise ValueError("npu_recurrent_kda: legacy q/k/v/g/beta ranks and token dimensions must match")
    warnings.warn(
        "recurrent_kda: rank3 without cu_seqlens and rank4 with cu_seqlens are deprecated "
        "shape aliases, supported through v26.12 and removed no earlier than v27.1. "
        "Use rank4 without cu_seqlens, or flatten to rank3 with cu_seqlens; layout is ignored.",
        DeprecationWarning, stacklevel=3,
    )
    output_shape = tuple(v.shape)
    if rank == 3:
        normalized = tuple(t.unsqueeze(0) for t in tensors)
    else:
        normalized = tuple(t.reshape(-1, *t.shape[2:]) for t in tensors)
    return (*normalized, output_shape)
