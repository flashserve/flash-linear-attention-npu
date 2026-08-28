# GDN Core Ablation

Every fusion step must compare against the same six-operator AscendC baseline:

`local_cumsum -> KKT -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o`

This is the complete GDN forward **core** scope used for formal Phase-to-Phase
production comparisons. It is not the full Demo/model path: causal convolution,
RMSNorm, and the output gate are outside this benchmark.

The benchmark records bit-exact output checks, host-side ACLNN call count, NPU
event latency, peak allocated NPU memory, per-ACLNN workspace sizes, and optional
device kernel count from a profiler trace.

```bash
TEST_DEVICE_ID=2 python torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py \
  --batch 1 --key-heads 4 --value-heads 8 --tokens 1024 \
  --chunk-size 64 --dtype bf16 --warmup 3 --iterations 10 --profile \
  --output test_output/gdn_ablation/dense_bf16_c64.json
```

For variable-length input, add canonical cumulative sequence lengths:

```bash
--cu-seqlens 0,384,1024
```

Do not compare invalid tail-padding columns in `A`; only the lower-triangular
region belonging to real tokens is part of the numerical contract. Keep all
other outputs bit exact. Add future fused implementations as named variants in
the same script rather than creating independent benchmarks.

Permanent one-ACLNN checkpoints are named
`phase1_one_aclnn_six_kernels` and
`phase2_one_aclnn_fused_kkt_solve`, and
`phase3_one_aclnn_fused_cumsum_kkt`, and
`phase4_one_aclnn_fused_fwd_ho`, and
`phase5_one_aclnn_fused_recompute_wu_ho`, and
`phase6_one_aclnn_fused_core`. The Phase 3 variant name is retained for
report compatibility, but its final route is cumulative
`ChunkCumsumKktSolveTri` (`cumsum + KKT + solve_tri`), not the rejected split
`ChunkCumsumKkt -> Cast -> SolveTri` candidate. The unversioned `composite_one_aclnn`
tracks the current default and must not be used as the only historical A/B.

Use `--phase4-accuracy-only` to compare Phase 4 directly with immutable Phase 3,
use `--phase4-paired-only` for the primary same-process performance comparison,
or run either checkpoint in an independent process with `--standalone-variant`.
The paired mode alternates Phase 3/4 launch order, reports both per-variant
latency summaries and per-iteration deltas, and records each checkpoint's
workspace in the same report:

```bash
python torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py \
  --device 1 --dtype fp16 --batch 1 --key-heads 4 --value-heads 8 \
  --tokens 1025 --chunk-size 64 --phase4-accuracy-only \
  --output test_output/gdn_ablation/phase4_accuracy.json

python torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py \
  --device 1 --dtype fp16 --batch 1 --key-heads 4 --value-heads 8 \
  --tokens 1025 --chunk-size 64 --phase4-paired-only \
  --warmup 20 --iterations 200 \
  --output test_output/gdn_ablation/phase4_paired.json

python torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py \
  --device 1 --dtype fp16 --batch 1 --key-heads 4 --value-heads 8 \
  --tokens 1025 --chunk-size 64 --standalone-variant phase4_one_aclnn_fused_fwd_ho \
  --warmup 10 --iterations 50 --profile \
  --output test_output/gdn_ablation/phase4_standalone.json
```

For Phase 5, use `--phase5-accuracy-only` or `--phase5-paired-only` to compare
the accepted `D+E+F` fused checkpoint directly with immutable Phase 4. The
paired mode uses the same alternating in-process NPU Event method. An
independent profiler run can select
`phase5_one_aclnn_fused_recompute_wu_ho` with `--standalone-variant`.

For Phase 6, use `--phase6-accuracy-only` to compare all public outputs directly
with immutable Phase 5, or `--phase6-paired-only` for same-process alternating
NPU Event comparison. The validated A2 matrix is dense and canonical varlen,
FP16/BF16 x C64/C128, `K=V=128`, with dense `T=1025/1024` and varlen `T=259`.
Use the same `--cu-seqlens` contract when reproducing a varlen point; this is a
versioned A/B API, not a promise that the unversioned default has switched:

```bash
python torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py \
  --device 7 --dtype fp16 --batch 1 --key-heads 4 --value-heads 8 \
  --tokens 1025 --chunk-size 64 --phase6-paired-only \
  --warmup 20 --iterations 200 \
  --output test_output/gdn_ablation/phase6_paired.json
```

Use `--all-phase-paired-only` for the archived direct Phase 0/1/2/3/4 comparison on one
input in one process. Every measured round runs all five variants. The base
order rotates so every variant occupies every position, then reverses after a
complete rotation. Use warmup and iteration counts divisible by ten so the
forward and reverse order cycles are balanced. This mode requires
a Phase 4 implementation that is live for the selected contract; do not use an
archived full-core-barrier artifact that is known to hang.

For varlen evidence only, add `--all-phase-skip-phase1` when repeated execution
of the historical Phase 1 variant reproduces its known device fault before the
new candidate runs. This keeps Phase 0/2/3/4 on the same input and in one
process; the forward/reverse order cycle is then eight rounds, so both warmup
and iteration counts must be divisible by eight. Record the omitted historical
variant explicitly rather than presenting the result as a five-variant matrix.

```bash
python torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py \
  --device 2 --dtype fp16 --batch 1 --key-heads 4 --value-heads 8 \
  --tokens 1024 --chunk-size 64 --all-phase-paired-only \
  --warmup 10 --iterations 50 \
  --output test_output/gdn_ablation/all_phase_dense_fp16_c64.json
```

Phase 2 also adds the five-call `fused_kkt_solve` variant. It replaces only
`chunk_scaled_dot_kkt -> solve_tri`; local cumsum, layout handling and the four
following GDN stages stay unchanged. The first fused kernel accepts the
Phase 1 physical contract where q/k have already been expanded to `Hv` heads.
