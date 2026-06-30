# ChunkLocalCumsum 精度验证报告

## 最新测试摘要

| 项目 | 信息 |
| --- | --- |
| 更新时间 | 2026-06-24 02:55:00 UTC |
| 测试平台 | Ascend 910B |
| CANN 路径 | `/usr/local/Ascend/cann-9.0.0` |
| 算子产物 | FLA 仓 `build/_CPack_Packages/Linux/External/*/packages/vendors/custom_transformer` 动态发现 |
| ACLNN 测试程序 | `/home/m00913889/codex04/flash-linear-attention-npu/build/chunk_local_cumsum_precision/test_aclnn_chunk_local_cumsum` |
| 接口状态 | 已对齐 Triton `chunk_local_cumsum` 核心入参 |
| 测试数据目录 | `/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/testdata` |
| 测试数据文件数 | 74 |
| case 总数 | 30 |
| 通过 case | 30 |
| 失败 case | 0 |
| 最大 `max_abs` | 0.000001907 |
| 最大 `bad_count` | 0 |
| 结论 | AscendC 输出与 Triton 标杆精度一致 |

## 标杆来源

- Triton 标杆脚本: `examples/generate_triton_cumsum_cases.py`
- 源 Triton 算子: `/home/m00913889/codex04/flash-linear-attention-npu/triton_ops/triton_core/cumsum.py`
- 标杆数据目录: `examples/testdata`
- 对拍入口: `examples/run_precision_test.sh`

Triton 标杆生成后会先和 PyTorch CPU reference 对比；只有 Triton 输出满足
`atol=3e-5, rtol=3e-5` 时才保存为 AscendC 对拍 golden。

## 运行命令

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
SKIP_TRITON_GENERATE=1 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh
```

## 测试覆盖

本轮共覆盖 30 个 case，包含固定长度和 varlen、正向和反向、带 scale 和不带 scale、`chunk_size=16/32/64`、多 batch、尾块、单 chunk、多 block、不同 H 维度。

| case | shape | chunk_size | reverse | scale | varlen |
| --- | --- | --- | --- | --- | --- |
| fixed_prefix_c64 | [1, 64, 32] | 64 | false | none | false |
| fixed_reverse_c64 | [1, 128, 32] | 64 | true | none | false |
| fixed_scale_batch2 | [2, 128, 32] | 64 | false | 0.25 | false |
| fixed_reverse_scale_c32 | [1, 96, 64] | 32 | true | -0.5 | false |
| fixed_tiny_c16 | [1, 17, 32] | 16 | false | none | false |
| fixed_reverse_tail_c16 | [1, 73, 64] | 16 | true | none | false |
| fixed_scale_neg_c16 | [2, 95, 32] | 16 | false | -1.25 | false |
| fixed_batch3_c32 | [3, 65, 64] | 32 | false | none | false |
| fixed_reverse_batch2_c32 | [2, 127, 64] | 32 | true | none | false |
| fixed_h128_c32 | [1, 96, 128] | 32 | false | none | false |
| fixed_reverse_h128_c32 | [1, 80, 128] | 32 | true | none | false |
| fixed_h96_c16 | [1, 129, 96] | 16 | false | none | false |
| fixed_reverse_h96_c16 | [2, 130, 96] | 16 | true | none | false |
| fixed_h16_c64 | [1, 129, 16] | 64 | false | none | false |
| fixed_reverse_h16_c64 | [2, 191, 16] | 64 | true | none | false |
| fixed_h8_c64 | [3, 257, 8] | 64 | false | none | false |
| fixed_reverse_h8_c64 | [1, 255, 8] | 64 | true | none | false |
| fixed_scale_small_input | [1, 128, 32] | 64 | false | 3.0 | false |
| fixed_reverse_scale_large_input | [1, 128, 32] | 64 | true | -0.75 | false |
| fixed_tail_c32_h32 | [2, 97, 32] | 32 | false | none | false |
| fixed_reverse_tail_c32_h32 | [1, 193, 32] | 32 | true | none | false |
| fixed_single_chunk_c64_b2 | [2, 31, 32] | 64 | false | none | false |
| fixed_reverse_single_chunk_c64 | [1, 63, 32] | 64 | true | none | false |
| fixed_exact_two_blocks_c64 | [2, 128, 32] | 64 | false | none | false |
| varlen_prefix_c64 | [1, 128, 32] | 64 | false | none | true |
| varlen_reverse_scale | [1, 128, 32] | 64 | true | 0.25 | true |
| varlen_tail_c32_h64 | [1, 113, 64] | 32 | false | none | true |
| varlen_reverse_tail_c32_h64 | [1, 113, 64] | 32 | true | none | true |
| varlen_c16_h96 | [1, 201, 96] | 16 | false | -0.25 | true |
| varlen_reverse_c16_h96 | [1, 201, 96] | 16 | true | 0.5 | true |

## 对拍结果

AscendC 与 Triton golden 对拍阈值: `atol=2e-5`, `rtol=2e-5`

| case | max_abs | max_rel | bad_count |
| --- | --- | --- | --- |
| fixed_prefix_c64 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_c64 | 0.000001431 | 0.000001431 | 0 |
| fixed_scale_batch2 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_scale_c32 | 0.000000477 | 0.000000221 | 0 |
| fixed_tiny_c16 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_tail_c16 | 0.000000477 | 0.000000477 | 0 |
| fixed_scale_neg_c16 | 0.000000000 | 0.000000000 | 0 |
| fixed_batch3_c32 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_batch2_c32 | 0.000001192 | 0.000001192 | 0 |
| fixed_h128_c32 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_h128_c32 | 0.000000715 | 0.000000715 | 0 |
| fixed_h96_c16 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_h96_c16 | 0.000000715 | 0.000000715 | 0 |
| fixed_h16_c64 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_h16_c64 | 0.000001431 | 0.000000447 | 0 |
| fixed_h8_c64 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_h8_c64 | 0.000001207 | 0.000001207 | 0 |
| fixed_scale_small_input | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_scale_large_input | 0.000001907 | 0.000000455 | 0 |
| fixed_tail_c32_h32 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_tail_c32_h32 | 0.000000954 | 0.000000336 | 0 |
| fixed_single_chunk_c64_b2 | 0.000000000 | 0.000000000 | 0 |
| fixed_reverse_single_chunk_c64 | 0.000000834 | 0.000000834 | 0 |
| fixed_exact_two_blocks_c64 | 0.000000000 | 0.000000000 | 0 |
| varlen_prefix_c64 | 0.000000000 | 0.000000000 | 0 |
| varlen_reverse_scale | 0.000000358 | 0.000000305 | 0 |
| varlen_tail_c32_h64 | 0.000000000 | 0.000000000 | 0 |
| varlen_reverse_tail_c32_h64 | 0.000000954 | 0.000000301 | 0 |
| varlen_c16_h96 | 0.000000000 | 0.000000000 | 0 |
| varlen_reverse_c16_h96 | 0.000000238 | 0.000000238 | 0 |

结论: 30/30 case 通过，AscendC 输出与 Triton 标杆精度一致。

## 修复记录

- 固定长度路径的 optional 输入用空 tensor 表示，tiling 通过元素数判断是否进入 varlen。
- `ASCEND_CUSTOM_OPP_PATH` 指向临时 vendor 目录 `.../vendors/custom_transformer`，确保运行时加载自定义 binary info。
- tiling 内直接从 `context->GetPlatformInfo()` 读取 AIV 核数，避免 ACLNN 动态路径 `compileInfo` 为空。
- 固定长度 kernel 改为按 16 个 `float` 一组的 cache-line 粒度分核，保证同一 64B GM cache line 只由一个 AIV core 写，避免多核写相邻小尾块时的 false sharing。
- varlen 路径当前采用保守单核调度，避免不对齐序列边界出现同类并发写冲突。
