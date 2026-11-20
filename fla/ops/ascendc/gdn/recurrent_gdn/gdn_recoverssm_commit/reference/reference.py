"""Independent CPU golden for ``gdn_recoverssm_commit``.

The reference walks the accepted replay records in reverse order and keeps
the same evaluation order as the Triton implementation in vLLM Ascend PR
20485.  It uses CPU float64 arithmetic and never imports the NPU runtime.
"""

from __future__ import annotations

import torch


def _cpu_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {tensor.device}")
    if any(step <= 0 for step in tensor.stride()):
        raise ValueError(f"{name} must have positive strides")
    return tensor.detach().to(device="cpu", dtype=torch.float64)


def gdn_recoverssm_commit_reference(
    state: torch.Tensor,
    replay: torch.Tensor,
    state_indices: torch.Tensor,
    commit_lens: torch.Tensor,
    final_state_indices: torch.Tensor,
    boundary_state_indices: torch.Tensor,
    boundary_recovery_lens: torch.Tensor,
    *,
    spec_query_len: int,
    align_mode: bool,
) -> torch.Tensor:
    """Return a committed copy of ``state``.

    ``state`` has logical shape ``[P, HV, V, K]`` and ``replay`` has shape
    ``[P, HV, L, V + K + 1]``.  All metadata vectors have one element per
    speculative request.  A non-positive page id is the null page and is
    skipped.
    """

    state64 = _cpu_tensor("state", state)
    replay64 = _cpu_tensor("replay", replay)
    source = _cpu_tensor("state_indices", state_indices).to(torch.int64)
    lengths = _cpu_tensor("commit_lens", commit_lens).to(torch.int64)
    final = _cpu_tensor("final_state_indices", final_state_indices).to(torch.int64)
    boundary = _cpu_tensor("boundary_state_indices", boundary_state_indices).to(
        torch.int64
    )
    recovery = _cpu_tensor("boundary_recovery_lens", boundary_recovery_lens).to(
        torch.int64
    )

    if state.ndim != 4:
        raise ValueError("state must have shape [P,HV,V,K]")
    if replay.ndim != 4:
        raise ValueError("replay must have shape [P,HV,L,V+K+1]")
    _, heads, value_dim, key_dim = state.shape
    if key_dim != value_dim or key_dim != 128:
        raise ValueError("this release supports K=V=128 only")
    if replay.shape[1:] != (
        heads,
        spec_query_len,
        value_dim + key_dim + 1,
    ):
        raise ValueError("replay shape does not match state and spec_query_len")
    for name, tensor in (
        ("state_indices", source),
        ("commit_lens", lengths),
        ("final_state_indices", final),
        ("boundary_state_indices", boundary),
        ("boundary_recovery_lens", recovery),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    rows = source.numel()
    if any(t.numel() != rows for t in (lengths, final, boundary, recovery)):
        raise ValueError("all commit metadata vectors must have the same length")
    if rows and (
        int(source.min()) < 0
        or int(source.max()) >= state.shape[0]
        or int(final.min()) < 0
        or int(final.max()) >= state.shape[0]
        or int(boundary.min()) < 0
        or int(boundary.max()) >= state.shape[0]
    ):
        raise ValueError("page ids must be null (0) or inside the state cache")

    committed = state64.clone()
    for row in range(rows):
        source_page = int(source[row])
        final_page = int(final[row])
        commit_len = min(max(int(lengths[row]), 0), spec_query_len)
        if source_page <= 0 or final_page <= 0 or commit_len == 0:
            continue

        boundary_page = int(boundary[row])
        recovery_len = min(int(recovery[row]), commit_len)
        need_boundary = (
            align_mode
            and boundary_page > 0
            and not (boundary_page == final_page and recovery_len == commit_len)
        )

        final_decay = torch.ones(
            heads, key_dim, dtype=torch.float64, device="cpu"
        )
        final_correction = torch.zeros(
            heads, value_dim, key_dim, dtype=torch.float64, device="cpu"
        )
        boundary_decay = final_decay.clone()
        boundary_correction = final_correction.clone()

        tokens = range(commit_len - 1, -1, -1)
        for position in tokens:
            record = replay64[source_page, :, position]
            correction = record[..., :value_dim]
            normalized_key = record[..., value_dim : value_dim + key_dim]
            decay = record[..., value_dim + key_dim]

            weighted_key = normalized_key * final_decay
            final_correction += correction[..., None] * weighted_key[:, None, :]
            final_decay *= decay[:, None]

            if need_boundary and position < recovery_len:
                boundary_key = normalized_key * boundary_decay
                boundary_correction += (
                    correction[..., None] * boundary_key[:, None, :]
                )
                boundary_decay *= decay[:, None]

        initial = state64[source_page]
        final_value = initial * final_decay[:, None, :] + final_correction
        if need_boundary:
            boundary_value = (
                initial * boundary_decay[:, None, :] + boundary_correction
            )
            committed[boundary_page] = boundary_value.to(committed.dtype)
        committed[final_page] = final_value.to(committed.dtype)

    return committed


__all__ = ["gdn_recoverssm_commit_reference"]
