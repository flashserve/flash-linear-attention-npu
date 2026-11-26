# compact_conv_state

Ascend 950 RecoverSSM convolution-history compaction operator.

The operator copies accepted convolution history from source cache pages into
final and optional boundary pages. Cache addresses and physical strides come
from a device-side per-layer table.

```python
from fla_npu.ops.ascendc import compact_conv_state

compact_conv_state(
    conv_state_ref,
    conv_state_base_addrs,
    conv_state_block_strides,
    conv_state_dim_strides,
    conv_state_token_strides,
    state_indices,
    commit_lens,
    final_state_indices,
    boundary_state_indices,
    boundary_recovery_lens,
    conv_history_len=3,
    align_mode=True,
    dim_stride=3,
)
```

The final and optional boundary histories are updated in place.
