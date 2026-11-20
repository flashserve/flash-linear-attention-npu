# gdn_recoverssm_commit

Ascend 950 checkpoint-commit kernel for Qwen GDN speculative decoding.

The operator consumes compact FP32 replay records and composes the accepted
prefix back into paged BF16/FP32 recurrent checkpoints. Replay is read-only.

```python
from fla_npu.ops.ascendc import gdn_recoverssm_commit

gdn_recoverssm_commit(
    state,
    replay,
    state_commit_cache_address_stride_table,
    state_indices,
    commit_lens,
    final_state_indices,
    boundary_state_indices,
    boundary_recovery_lens,
    spec_query_len=5,
    align_mode=True,
)
```

The public call returns `None` and updates `state` in place. See
[docs/api.md](docs/api.md) for the address/stride table, mathematical contract,
supported shapes, and page rules. The implementation design and UB/synchronization
budget are in [docs/design.md](docs/design.md).

Current scope:

- Ascend 950.
- `K=V=128`.
- `1 <= spec_query_len <= 32`.
- BF16 or FP32 checkpoint; FP32 replay.
- Multiple layers through the device-side address/stride table.
- Optional aligned boundary checkpoint commit.
