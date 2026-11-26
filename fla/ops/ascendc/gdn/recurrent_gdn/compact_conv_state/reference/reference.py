"""Exact reference for RecoverSSM convolution-history compaction."""


def compact_conv_state_reference(
        conv_state_views,
        state_indices,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        *,
        conv_history_len,
        align_mode,
        null_block_id=0):
    """Copy accepted histories in-place on logical ``[pages, C, slots]`` views."""

    source_values = [int(value) for value in state_indices]
    lengths = [int(value) for value in commit_lens]
    finals = [int(value) for value in final_state_indices]
    boundaries = [int(value) for value in boundary_state_indices]
    recovery_lengths = [int(value) for value in boundary_recovery_lens]

    for view in conv_state_views:
        for row, source_page in enumerate(source_values):
            commit_len = lengths[row]
            final_page = finals[row]
            if source_page <= null_block_id or final_page <= null_block_id or commit_len == 0:
                continue

            start = commit_len - 1
            final_values = view[
                source_page, :, start:start + conv_history_len
            ].clone()

            boundary_page = boundaries[row] if align_mode else null_block_id
            recovery_len = recovery_lengths[row] if align_mode else 0
            boundary_valid = (
                boundary_page > null_block_id
                and recovery_len > 0
                and not (
                    boundary_page == final_page
                    and recovery_len == commit_len
                )
            )
            boundary_values = None
            if boundary_valid:
                boundary_start = recovery_len - 1
                boundary_values = view[
                    source_page,
                    :,
                    boundary_start:boundary_start + conv_history_len,
                ].clone()

            if boundary_valid:
                view[
                    boundary_page, :, :conv_history_len
                ].copy_(boundary_values)
            view[final_page, :, :conv_history_len].copy_(final_values)

    return conv_state_views
