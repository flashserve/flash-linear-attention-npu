# ChunkLocalCumsum Triton vs AscendC 性能对比

## 摘要

- 数据来源: `msprof op`
- 指标: `Task Duration(us)`
- profiling 根目录: `/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/profiling/perf_final_after_opt1`
- 有效 case 数: 30/30
- 平均 speedup `triton_us / ascendc_us`: 0.533x

## 明细

| case | shape | chunk | reverse | scale | varlen | Triton us | AscendC us | speedup(T/A) | faster |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| fixed_prefix_c64 | [1, 64, 32] | 64 | false | 1.0 | false | 2.800 | 5.460 | 0.513 | triton |
| fixed_reverse_c64 | [1, 128, 32] | 64 | true | 1.0 | false | 3.220 | 6.200 | 0.519 | triton |
| fixed_scale_batch2 | [2, 128, 32] | 64 | false | 0.25 | false | 3.180 | 7.540 | 0.422 | triton |
| fixed_reverse_scale_c32 | [1, 96, 64] | 32 | true | -0.5 | false | 3.240 | 5.840 | 0.555 | triton |
| fixed_tiny_c16 | [1, 17, 32] | 16 | false | 1.0 | false | 2.880 | 4.600 | 0.626 | triton |
| fixed_reverse_tail_c16 | [1, 73, 64] | 16 | true | 1.0 | false | 3.820 | 5.320 | 0.718 | triton |
| fixed_scale_neg_c16 | [2, 95, 32] | 16 | false | -1.25 | false | 3.160 | 7.080 | 0.446 | triton |
| fixed_batch3_c32 | [3, 65, 64] | 32 | false | 1.0 | false | 3.200 | 6.120 | 0.523 | triton |
| fixed_reverse_batch2_c32 | [2, 127, 64] | 32 | true | 1.0 | false | 3.520 | 6.780 | 0.519 | triton |
| fixed_h128_c32 | [1, 96, 128] | 32 | false | 1.0 | false | 2.720 | 6.400 | 0.425 | triton |
| fixed_reverse_h128_c32 | [1, 80, 128] | 32 | true | 1.0 | false | 3.220 | 5.900 | 0.546 | triton |
| fixed_h96_c16 | [1, 129, 96] | 16 | false | 1.0 | false | 3.680 | 5.980 | 0.615 | triton |
| fixed_reverse_h96_c16 | [2, 130, 96] | 16 | true | 1.0 | false | 4.180 | 9.120 | 0.458 | triton |
| fixed_h16_c64 | [1, 129, 16] | 64 | false | 1.0 | false | 3.080 | 5.920 | 0.520 | triton |
| fixed_reverse_h16_c64 | [2, 191, 16] | 64 | true | 1.0 | false | 4.000 | 6.200 | 0.645 | triton |
| fixed_h8_c64 | [3, 257, 8] | 64 | false | 1.0 | false | 3.680 | 8.820 | 0.417 | triton |
| fixed_reverse_h8_c64 | [1, 255, 8] | 64 | true | 1.0 | false | 3.860 | 5.840 | 0.661 | triton |
| fixed_scale_small_input | [1, 128, 32] | 64 | false | 3.0 | false | 3.400 | 7.280 | 0.467 | triton |
| fixed_reverse_scale_large_input | [1, 128, 32] | 64 | true | -0.75 | false | 3.400 | 6.940 | 0.490 | triton |
| fixed_tail_c32_h32 | [2, 97, 32] | 32 | false | 1.0 | false | 2.980 | 5.720 | 0.521 | triton |
| fixed_reverse_tail_c32_h32 | [1, 193, 32] | 32 | true | 1.0 | false | 3.580 | 5.660 | 0.633 | triton |
| fixed_single_chunk_c64_b2 | [2, 31, 32] | 64 | false | 1.0 | false | 3.300 | 5.060 | 0.652 | triton |
| fixed_reverse_single_chunk_c64 | [1, 63, 32] | 64 | true | 1.0 | false | 3.600 | 5.520 | 0.652 | triton |
| fixed_exact_two_blocks_c64 | [2, 128, 32] | 64 | false | 1.0 | false | 3.320 | 6.020 | 0.551 | triton |
| varlen_prefix_c64 | [1, 128, 32] | 64 | false | 1.0 | true | 4.160 | 6.500 | 0.640 | triton |
| varlen_reverse_scale | [1, 128, 32] | 64 | true | 0.25 | true | 4.060 | 7.680 | 0.529 | triton |
| varlen_tail_c32_h64 | [1, 113, 64] | 32 | false | 1.0 | true | 3.500 | 7.780 | 0.450 | triton |
| varlen_reverse_tail_c32_h64 | [1, 113, 64] | 32 | true | 1.0 | true | 4.020 | 7.280 | 0.552 | triton |
| varlen_c16_h96 | [1, 201, 96] | 16 | false | -0.25 | true | 4.560 | 13.580 | 0.336 | triton |
| varlen_reverse_c16_h96 | [1, 201, 96] | 16 | true | 0.5 | true | 5.080 | 13.600 | 0.374 | triton |

## 原始数据

- JSON: `/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/performance_comparison.json`
- profiling: `/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/profiling/perf_final_after_opt1`
