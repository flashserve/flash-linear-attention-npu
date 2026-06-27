# Chunk KDA 算子与三方实现精度对比测试报告

## 1. 测试对象

待测对象：
- `npu_chunk_kda_fwd`
- `npu_chunk_kda_bwd`
- 增强后的 `npu_chunk_gated_delta_rule_fwd_h(gk=...)`
- 增强后的 `npu_chunk_gated_delta_rule_bwd_dhu(gk=...)`

对标对象：
- fla-org KDA 公式语义。
- 仓内 PyTorch reference：`tests/reference/chunk_kda_reference.py`。

说明：当前验证环境未提供可直接运行的 fla-org Triton GPU kernel，因此本报告用与 fla-org KDA 公式对齐的 PyTorch reference 做数值对标。

## 2. 当前实现状态

- `torch.ops.npu.npu_chunk_kda_fwd/bwd` 已作为 KDA 大融合 L2 入口落地。
- KDA forward 已拆为块内重算、跨 chunk 状态递推、输出合成三段：`K=128,V=128` 场景优先复用增强后的 native `fwd_h(gk)`，若 native L0 不可用或形状不满足约束则自动回退到 composite 状态递推。
- `chunk_size=128,V=256` 仍走 composite fallback，避免复用现有 `fwd_h` 的 V256 挂起路径；该 fallback 已通过 forward 精度测试。
- `chunk_gated_delta_rule_bwd_dhu` 已补齐 `gk` 分支，按三方 KDA 语义实现 `dv2 = kg @ dh + dv`，并在 `dh` 递推中按 K 维执行 `exp2(gk_last)` state decay。

## 3. 已执行测试

| Case | 覆盖范围 | 结果 |
|---|---|---|
| 构建 | `fwd_h + bwd_dhu` custom 包构建 | 通过 |
| 安装 | custom 包安装、`fla_npu` wheel 构建安装 | 通过 |
| KDA native hit | `K=128,V=128,chunk_size=64` 命中 native `fwd_h(gk)` | 通过 |
| KDA forward V256 | `T=128,K=128,V=256,chunk_size=128` forward vs reference | 通过 |
| KDA backward medium | `T=64,K=64,V=128,chunk_size=64` backward vs autograd golden | 通过 |
| KDA varlen/GVA | 仓内 KDA 测试脚本覆盖 varlen、GVA、V256 小形状 | 通过 |
| `bwd_dhu(gk)` precision | `T=128,K=128,V=128,chunk_size=64` vs CPU golden | 通过 |

## 4. 关键数值结果

| Case | 输出 | max_abs | mean_abs | 结果 |
|---|---|---:|---:|---|
| KDA native forward fp16 | `final_state` | `2.22e-6` | `3.51e-7` | 通过 |
| KDA forward V256 fp16 | `o` | `2.38e-7` | `1.64e-8` | 通过 |
| KDA forward V256 fp16 | `final_state` | `5.59e-9` | `1.81e-10` | 通过 |
| KDA backward medium fp16 | `dq/dk/dv/dh0` | `0` | `0` | 通过 |
| KDA backward medium fp16 | `dbeta/dgk` | `1.83e-3` | `7.54e-5` | 通过 |

阈值：
- fp16：`rtol=5e-2, atol=5e-2`
- bf16：`rtol=8e-2, atol=8e-2`

## 5. 结论

当前 PR 已具备一套可运行的 KDA 正反向大融合 L2 算子，功能和精度与公式级 reference 对齐。`K=128,V=128` forward 状态段已验证可复用 native `fwd_h(gk)`；`bwd_dhu(gk)` 已完成 native kernel 语义并通过精度测试；`chunk_size=128,V=256` forward 通过 composite fallback 保持功能覆盖。
