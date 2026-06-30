# ChunkLocalCumsum AscendC Optimization Report

## Scope

This pass implements the first optimization strategy: keep one whole chunk resident in UB, compute the local prefix/suffix sum in UB, then copy the whole chunk back to GM.

Changed file:

- `op_kernel/chunk_local_cumsum.cpp`

## Implementation

- Added a chunk-resident fast path for 32B-aligned `H`.
- Fast path allocates one UB chunk buffer sized as `chunk_size * align_up(min(H, 512), 8) * sizeof(float)`.
- Fast path performs:
  - one GM to UB chunk load when contiguous/aligned, otherwise multi-block `DataCopyPad`;
  - in-place chunk prefix/suffix scan in UB with vector `Add`;
  - optional in-place `Muls` only when `scale != 1.0`;
  - one UB to GM chunk store when contiguous/aligned, otherwise multi-block `DataCopyPad`.
- Non-32B-aligned `H` is gated back to the previous row-wise path. The `[1, 19, 12]` case exposed that multi-row `DataCopyPad` packed `H=12` rows tightly while the fast path computed with 16-float row stride, so the conservative gate keeps correctness and avoids a regression on this already-fast small case.

## Validation

Build:

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
```

Static check:

```bash
python3 /root/.codex/skills/op-dev/shared/scripts/check_ascendc_buffer_lifecycle.py \
  /home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/op_kernel
```

Precision:

| suite | result |
| --- | --- |
| 3 perf-review cases | PASS, `bad_count=0` |
| 30 generated regression cases | PASS, max `max_abs=1.907e-6` |
| `H=513` edge case | PASS, `bad_count=0` |

## Performance

Profile output:

- JSON: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.json`
- Markdown: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.md`
- 3-case raw profiles: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/profiling/perf_review_after_chunk_fast`
- 30-case raw profiles: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/profiling/perf_30cases_after_chunk_fast`

| case | shape | Triton us | AscendC before us | AscendC after us | after vs before | final faster |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `fixed_shape_19_12_c16` | `[1, 19, 12]` | 26.680 | 6.680 | 6.520 | 1.025x | AscendC 4.09x |
| `fixed_prefix_c64` | `[1, 64, 32]` | 2.840 | 15.280 | 6.020 | 2.538x | Triton 2.12x |
| `fgd_g_64_128_512_c16` | `[64, 128, 512]` | 33.640 | 76.380 | 34.140 | 2.237x | Triton 1.015x |

30-case comparison after the first optimization:

| group | cases | avg Triton us | avg AscendC us | avg `Triton/AscendC` | faster distribution |
| --- | ---: | ---: | ---: | ---: | --- |
| all | 30 | 3.575 | 6.832 | 0.540x | Triton 30, AscendC 0 |
| fixed | 24 | 3.434 | 6.210 | 0.558x | Triton 24, AscendC 0 |
| varlen | 6 | 4.140 | 9.320 | 0.468x | Triton 6, AscendC 0 |

Worst relative gaps in the 30-case suite:

| case | shape | Triton us | AscendC us | `Triton/AscendC` |
| --- | --- | ---: | ---: | ---: |
| `varlen_c16_h96` | `[1, 201, 96]` | 4.360 | 13.020 | 0.335x |
| `varlen_reverse_c16_h96` | `[1, 201, 96]` | 5.160 | 13.060 | 0.395x |
| `fixed_scale_small_input` | `[1, 128, 32]` | 2.980 | 7.140 | 0.417x |

## Remaining Gaps

- `[1, 64, 32]` still trails Triton because it has only one chunk/H tile task. Runtime is dominated by kernel launch, queue setup, and 63 serial vector add/barrier steps; there is almost no parallelism to amortize AscendC framework overhead.
- `[64, 128, 512]` is now effectively at parity, but still about 1.5% slower in this run. The remaining cost is mainly serial per-row prefix accumulation inside each chunk plus synchronization between vector compute and MTE3 store.
- A second optimization should replace the serial row-by-row scan with a log-step in-UB prefix scan using an additional UB workspace, reducing barriers from `chunk_len - 1` to `ceil(log2(chunk_len))`. This is most relevant for `chunk_size=16/64` aligned-H cases.

## 2026-06-24 Supplemental Review

Kept change:

- `[opt-1]` Host tiling now rejects `head_first=true` instead of silently applying `[B, T, H]` addressing. The current AscendC kernel only implements `[B, T, H]`, so this is a semantic safety fix and does not touch the kernel hot path.

Rejected experiment:

- `[opt-2]` Changed fast-path scale from per-row `Muls` to one whole-chunk `Muls` when rows are tightly packed. Precision passed, but performance regressed and the change was reverted.

Validation:

| step | result |
| --- | --- |
| `[opt-1]` after-change Triton golden comparison | 30/30 PASS, all `bad_count=0` |
| `[opt-2]` experiment Triton golden comparison | 30/30 PASS, all `bad_count=0` |
| final code after reverting `[opt-2]` | 30/30 PASS, all `bad_count=0`, max `max_abs=1.907e-6` |
| buffer lifecycle scan | PASS, no missing `InitBuffer` |

Performance decision:

| version | group | avg AscendC us | avg `Triton/AscendC` | decision |
| --- | --- | ---: | ---: | --- |
| first chunk-fast baseline | all 30 | 6.832 | 0.540x | baseline |
| `[opt-2]` experiment | all 30 | 6.883 | 0.534x | reverted |
| `[opt-2]` experiment | scale cases | 8.628 | - | reverted |
| final `[opt-1]` only run | all 30 | 6.925 | 0.533x | kernel hot path unchanged; single-sample profiling drift |

Final profiling output:

- JSON: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.json`
- Markdown: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.md`
- Raw profiles: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/profiling/perf_final_after_opt1`
