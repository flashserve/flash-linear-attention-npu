# Qwen sparse attention (Ascend 950PR, CATLASS Python DSL)

Experimental example of a paged, token-selected sparse attention kernel written in the
CATLASS Python DSL. It is **not** a production operator and is not enabled in any model
path.

## Files

- `qwen_sparse_attention.py` — sparse kernel used for the reported correctness result.
  Paged KV cache, per-row sorted `logical_indices`, GQA (one KV head), online softmax.
- `qwen_dense_page.py` — **WIP** page-level variant (see semantics below). Not validated.
- `DESIGN.md` — design, measured bottleneck and honest performance analysis.
- `bench_r3h_summary.md` — 50-case benchmark against the task's derived baseline.

## Status

- Sparse kernel correctness: passes the task reference within `max_abs = 1.56e-2`.
- Performance: `0/50` cases reach `1.0x` of the task baseline; median `0.210x`,
  geomean `0.146x`. The baseline is derived as `dense FIA latency x KV density` and
  assumes selected-KV gather costs nothing; large-context cases would require
  13-175 TB/s gather bandwidth (see DESIGN).
- Page-level variant: changes the operator semantics to 16-token pages and drops the
  token-count mask. Its accuracy and timing have **not** been validated yet; do not merge.

## Requirements

- Ascend 950PR (`--npu-arch 3510`), CANN 9.2.0
- `ascend-catlass-dsl 2.2.1.dev20261112` (or compatible)
- PyTorch + torch_npu on the host

## Notes for review

The sparse kernel is DMA-issue bound: one 512 B DMA per selected token per K/V.
16-token page-tile copies were validated in isolation, but a correct token-count mask
could not be built with the available DSL primitives in the time budget. This PR is a
draft to preserve the implementation and measurements for discussion; it should not be
merged without an owner and a validation plan.
