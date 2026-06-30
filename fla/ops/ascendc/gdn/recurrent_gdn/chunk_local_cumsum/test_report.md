# ChunkLocalCumsum 测试报告

## 测试摘要

| 项目 | 信息 |
| --- | --- |
| 更新时间 | 2026-06-24 02:55:00 UTC |
| 测试平台 | Ascend 910B |
| CANN 路径 | `/usr/local/Ascend/cann-9.0.0` |
| 算子产物 | FLA 仓 `build/_CPack_Packages/Linux/External/*/packages/vendors/custom_transformer` 动态发现 |
| ACLNN 测试程序 | `/home/m00913889/codex04/flash-linear-attention-npu/build/chunk_local_cumsum_precision/test_aclnn_chunk_local_cumsum` |
| 接口状态 | 已对齐 Triton `chunk_local_cumsum` 核心入参，移除 `has_scale`，新增 `head_first/output_dtype` |
| Triton golden 脚本 | `/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/generate_triton_cumsum_cases.py` |
| 对拍脚本 | `/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh` |
| 测试数据目录 | `/home/m00913889/codex04/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/testdata` |
| 测试数据文件数 | 74 |
| case 总数 | 30 |
| 通过 case | 30 |
| 失败 case | 0 |
| 对拍阈值 | `atol=2e-5`, `rtol=2e-5` |
| 最大 `max_abs` | 0.000001907 |
| 最大 `bad_count` | 0 |
| 结论 | 通过 |

## 执行命令

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
SKIP_TRITON_GENERATE=1 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh
```

## 测试流程

1. 使用 Triton 原始算子生成 golden。
2. Triton 输出先与 PyTorch CPU reference 校验，阈值为 `atol=3e-5`, `rtol=3e-5`。
3. 保存输入、Triton golden、varlen 辅助输入和 `cases.txt` 到 `examples/testdata`。
4. 编译并运行 ACLNN C++ 测试程序。
5. 将 AscendC 输出拷回 host，与 Triton golden 逐元素对拍。

## 覆盖范围

- fixed 和 varlen
- `reverse=false` 和 `reverse=true`
- 带 scale 和不带 scale
- `head_first=false`
- `output_dtype=float32`
- `chunk_size=16/32/64`
- 多 batch
- 单 chunk、多 block、tail block
- `H=8/16/32/64/96/128`

## 结果明细

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

## 关联报告

- `precision_report.md`: 精度验证详情。
- `operator_description.md`: 算子功能、输入输出和实现逻辑。
- `modification_record.md`: 迁移和修复记录。
