"""Independent CPU merge from docs/design.md's source-rank recurrence."""
import torch


def merge_fwd_bwd_kernel_cpu(h, ag_hm, pre_or_post_num_ranks, rank,
                             forward=True, state_v_first=False):
    count = int(pre_or_post_num_ranks)
    world = int(ag_hm.shape[0])
    state = None
    for step in range(count):
        source = (0 if world == 1 else int(rank) - count + step
                  if forward else int(rank) + count - step)
        pair = ag_hm[source].to(torch.float64)
        he, matrix = pair[..., :128], pair[..., 128:]
        state = he.clone() if state is None else matrix @ state + he
    if state_v_first:
        state = state.transpose(-1, -2)
    # Match the public in-place result, including its alias and dtype.
    h.copy_(state.to(h.dtype))
    return h
