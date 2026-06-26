# Chunk KDA 算子与三方实现精度对比测试报告

## 1. 测试对象

待测对象：

- `npu_chunk_kda_fwd`
- `npu_chunk_kda_bwd`
- 增强后的 `npu_chunk_gated_delta_rule_fwd_h(gk=...)`

对标对象：

- fla-org KDA 公式语义。
- 仓内 PyTorch reference：`tests/reference/chunk_kda_reference.py`。

说明：当前验证环境未提供可直接运行的 fla-org Triton GPU kernel，因此本报告用与 fla-org KDA 公式对齐的 PyTorch reference 做数值对标。

## 2. 当前实现状态

- 已新增 `torch.ops.npu.npu_chunk_kda_fwd` 和 `torch.ops.npu.npu_chunk_kda_bwd`。
- 当前 KDA 正反向为 PyTorch/NPU composite core，用于固定 ABI、打通功能和精度闭环；后续 Ascend C 大融合 L0/kernel 可在不改变 Python ABI 的前提下替换内部实现。
- 已新增纯 PyTorch KDA forward reference。
- 已增强 `chunk_gated_delta_rule_fwd_h` 的 `gk` 下发和 kernel 消费路径，`gk` 分支按 K 维 `exp2(gk_last)` 做 state decay；原 `g` 分支保持自然指数路径。
- 已发现现有 `chunk_gated_delta_rule_fwd_h` 在 `chunk_size=128,V=256` 形状下原 `g` 分支也会运行挂起，因此 KDA 大融合不能直接复用该状态核覆盖 V256，需按 V tile 重新设计状态更新 pipeline。

## 3. 已执行测试

| Case | 覆盖范围 | 结果 |
|---|---|---|
| 构建 | `chunk_gated_delta_rule_fwd_h` custom 包构建 | 通过 |
| 安装 | custom 包安装、`fla_npu` wheel 构建安装 | 通过 |
| 注册 | `npu_chunk_kda_fwd/bwd` 注册检查 | 通过 |
| KDA forward smoke | 小形状 forward vs reference | 通过 |
| KDA forward V256 | `T=128,K=128,V=256,chunk_size=128` forward vs reference | 通过 |
| KDA backward medium | `T=64,K=64,V=128,chunk_size=64` backward vs autograd golden | 通过 |
| KDA varlen/GVA | 仓内 KDA 测试脚本覆盖 varlen、GVA、V256 小形状 | 通过 |
| `fwd_h(gk)` precision | `T=128,K=128,V=128,chunk_size=64` vs CPU golden | 通过 |
| `fwd_h` V256 probe | `T=256,K=128,V=256,chunk_size=128` | 未通过，运行阶段挂起 |

## 4. 关键数值结果

| Case | 输出 | max_abs | mean_abs | 结果 |
|---|---|---:|---:|---|
| `fwd_h(gk)` fp16 | `h` | `1.53e-5` | `6.18e-7` | 通过 |
| `fwd_h(gk)` fp16 | `v_new` | `6.10e-5` | `5.78e-7` | 通过 |
| `fwd_h(gk)` fp16 | `final_state` | `7.69e-6` | `8.17e-7` | 通过 |
| KDA forward V256 fp16 | `o` | `2.38e-7` | `1.64e-8` | 通过 |
| KDA forward V256 fp16 | `final_state` | `5.59e-9` | `1.81e-10` | 通过 |
| KDA backward medium fp16 | `dq/dk/dv/dh0` | `0` | `0` | 通过 |
| KDA backward medium fp16 | `dbeta/dgk` | `1.83e-3` | `7.54e-5` | 通过 |

阈值：

- fp16：`rtol=5e-2, atol=5e-2`
- bf16：`rtol=8e-2, atol=8e-2`

## 5. 当前结论

当前 PR 已具备一套可运行的 KDA 正反向 composite 算子，功能和精度与公式级 reference 对齐，并覆盖了 `chunk_size=128,V=256` 的 forward 场景以及中等规模 backward 梯度场景。

`chunk_gated_delta_rule_fwd_h(gk)` 已完成可用的 K/V=128 路径并通过精度验证；但 `chunk_size=128,V=256` 不能直接复用现有 `fwd_h` kernel。后续大融合 KDA kernel 需要将 V256 状态更新按 V tile 驻留 UB/L1 重新设计，避免沿用当前 `fwd_h` 的 V256 挂起路径。

## 6. 后续验证要求

- 为 KDA 大融合新增 L0 后补齐 ATK generator/executor/yaml/json。
- 对新增 Ascend C kernel 执行 ATK 精度验证。
- 对新增或增强的 Ascend C kernel 执行 sanitizer 验证，按问题类型覆盖 racecheck、memcheck、initcheck、synccheck。
- 对 `chunk_size=128,V=256` 的大融合状态更新路径增加单独回归用例。
