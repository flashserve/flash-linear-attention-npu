# aclnnChunkKktSolveTri

`aclnnChunkKktSolveTri` is the Phase 2 fused GDN stage:

```text
chunk_scaled_dot_kkt -> solve_tri
```

It removes the public fp32 KKT intermediate and the second kernel launch while
preserving the solved `A` contract used by `recompute_w_u`. The first version
keeps a private typed KKT hand-off buffer in the ACLNN workspace. Removing that
GM hand-off requires a global stage barrier or a tile-local pipeline; a simple
in-place alias is unsafe for multi-core varlen execution.

## Contract

```text
k:             [B, H, T, 128], fp16/bf16
g:             [B, H, T], fp32, local cumulative gate
beta:          [B, H, T], fp32
cu_seqlens:    optional int64 array
chunk_indices: optional flat int64 [sequence, local_chunk] array
chunk_size:    64 or 128
A:             [B, H, T, chunk_size], same dtype as k
```

Variable-length input uses physical `B=1` and requires both metadata arrays in
canonical sequence-major order. The first Phase 2 implementation keeps the
Phase 1 physical head contract: q/k must already be expanded to the value-head
count before this stage. Native `Hk != Hv` handling is deferred.

Only valid lower-triangular regions in tail chunks are mathematically
meaningful. The implementation nevertheless zero-fills padded columns for all
valid token rows because `recompute_w_u` consumes the full row width. Tests
therefore compare every valid token row bit-for-bit against the existing
two-kernel path, covering dense/varlen, FP16/BF16, multiple batch/head sizes,
tail chunks and `chunk_size=64/128`.
