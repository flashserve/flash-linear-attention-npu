# Chunk KDA 算子与三方实现精度对比测试报告

## 1. 测试对象

待测对象：
- `npu_chunk_kda_fwd`
- `npu_chunk_kda_bwd`
- 增强后的 `npu_chunk_gated_delta_rule_fwd_h(gk=...)`
- 增强后的 `npu_chunk_gated_delta_rule_bwd_dhu(gK=...)`

对标对象：
- fla-org KDA PyTorch/Triton 语义参考。
- 单算子 ATK golden 使用等价 PyTorch 公式，不依赖 GPU Triton kernel。

## 2. 当前状态

截至本文档归档时：
- 已在 `fla_npu` 包内新增完整 KDA composite operator：`torch.ops.npu.npu_chunk_kda_fwd` 和 `torch.ops.npu.npu_chunk_kda_bwd`。
- 当前实现为 PyTorch/NPU composite core，用于先打通公开 ABI、正反向功能和精度闭环；Ascend C 大融合 L0/kernel 替换仍按设计文档推进。
- 已新增纯 PyTorch KDA forward 参考实现 `tests/reference/chunk_kda_reference.py`，作为公式级三方语义 golden。
- 已执行参考实现和 `fla_npu` KDA 测试脚本语法检查，结果通过。
- 已执行 `fla_npu` wheel 构建、安装和 KDA op 注册检查，结果通过。
- 已在 NPU 环境执行 KDA forward/backward 精度测试，覆盖 `chunk_size=64`、`chunk_size=128`、`V=256`、GVA 和 varlen 场景，结果通过，未发现 NaN/Inf。
- 当前验证环境未安装 fla-org `fla` 与 Triton 包，且三方 KDA Triton kernel 依赖 GPU Triton 运行环境；因此未执行三方 kernel 直接运行对比。本报告使用与 fla-org KDA 公式一致的 PyTorch reference 做数值对标。
- 尚未执行 KDA Ascend C 大融合 kernel 的 ATK 精度和 sanitizer 验证。

## 3. 计划测试矩阵

| Case | B | H | HV | T | K | V | chunk_size | dtype | varlen | 目标 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FWD-SMOKE | 1 | 4 | 4 | 2048 | 128 | 128 | 64 | fp16 | 否 | 前向冒烟 |
| FWD-BF16 | 1 | 8 | 8 | 4096 | 128 | 128 | 64 | bf16 | 否 | bf16 常规 |
| FWD-LONG | 1 | 16 | 16 | 8192 | 128 | 128 | 128 | bf16 | 否 | 长序列 |
| FWD-V256 | 1 | 16 | 16 | 4096 | 128 | 256 | 128 | bf16 | 否 | Vdim 256 |
| FWD-GVA | 1 | 8 | 16 | 4096 | 128 | 128 | 128 | bf16 | 否 | GVA |
| FWD-VARLEN | 1 | 8 | 8 | sum(lens) | 128 | 128 | 64/128 | bf16 | 是 | varlen |
| BWD-SMOKE | 1 | 4 | 4 | 2048 | 128 | 128 | 64 | fp16 | 否 | 反向冒烟 |
| BWD-V256 | 1 | 16 | 16 | 4096 | 128 | 256 | 128 | bf16 | 否 | Vdim 256 反向 |
| BWD-VARLEN | 1 | 8 | 8 | sum(lens) | 128 | 128 | 64/128 | bf16 | 是 | varlen 反向 |

## 4. 验收指标

前向输出：
- `o`
- `final_state`
- `Aqk`
- `Akk`

反向输出：
- `dq`
- `dk`
- `dv`
- `dbeta`
- `dgk`
- `dh0`

阈值：

| dtype | rtol | atol |
|---|---:|---:|
| fp16 | 3e-2 | 3e-2 |
| bf16 | 5e-2 | 5e-2 |
| fp32 中间梯度 | 1e-3 | 1e-3 |

同时记录：
- `max_abs`
- `max_rel`
- `mean_abs`
- cosine similarity
- NaN/Inf 数量

## 5. ATK 验证口径

每个新增或增强算子均需提供：
- `generator_*.py`
- `executor_*.py`
- `aclnn_*.yaml`
- `all_aclnn_*.json`

执行项：

```bash
atk node --backend PYACLNN node --backend CPU task -c all_aclnn_chunk_kda_fwd.json --task accuracy -p executor_chunk_kda_fwd.py
atk node --backend PYACLNN node --backend CPU task -c all_aclnn_chunk_kda_bwd.json --task accuracy -p executor_chunk_kda_bwd.py
```

说明：
- 命令示例不包含设备编号、服务器名、绝对路径和日志路径。
- 实测报告只记录测试项和结果，不记录内部环境信息。

## 6. Sanitizer 验证口径

待测 kernel：
- `chunk_kda_fwd_intra_recompute`
- `chunk_kda_fwd_o_gk`
- `chunk_kda_bwd_dav`
- `chunk_kda_bwd_wy_dqkg_fused`
- `chunk_kda_bwd_intra`
- `chunk_gated_delta_rule_fwd_h` 的 `gk` 分支
- `chunk_gated_delta_rule_bwd_dhu` 的 `gk` 分支

工具：
- race 类：`mssanitizer --tool=racecheck`
- 越界类：`memcheck`
- 未初始化读取：`initcheck`
- 同步类：`synccheck`

结论要求：
- 确认实际加载 sanitizer 版本对象。
- 保留原始日志和聚合摘要。
- 区分真实代码问题和工具对手工流水/非 TPipe 管理内存的保守报告。

## 7. 实测结果

| Case | 输出 | max_abs | max_rel | mean_abs | cosine | NaN/Inf | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| BUILD | fla_npu wheel | - | - | - | - | - | 通过 |
| REGISTRY | npu_chunk_kda_fwd/bwd | - | - | - | - | - | 通过 |
| REF-SMOKE | o | - | - | - | - | 0 | 通过 |
| REF-SMOKE | final_state | - | - | - | - | 0 | 通过 |
| REF-C128-V256 | o | - | - | - | - | 0 | 通过 |
| REF-GVA | o | - | - | - | - | 0 | 通过 |
| REF-VARLEN | o | - | - | - | - | 0 | 通过 |
| FWD-SMOKE | o | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| FWD-SMOKE | final_state | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| FWD-V256-GVA-VARLEN | o | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| FWD-V256-GVA-VARLEN | final_state | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| BWD-SMOKE | dq | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| BWD-SMOKE | dk | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| BWD-SMOKE | dv | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| BWD-SMOKE | dbeta | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| BWD-SMOKE | dgk | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |
| BWD-SMOKE | dh0 | 阈值内 | 阈值内 | 阈值内 | - | 0 | 通过 |

## 8. 当前结论

当前已完成 KDA composite operator 的正反向功能、NPU 注册和小型精度闭环验证。该结论仅覆盖 PyTorch/NPU composite core，不等价于 Ascend C 大融合 kernel 的性能或 sanitizer 结论；后续将按设计文档继续替换为大融合 L0/kernel，并补充 ATK 与 sanitizer 结果。
