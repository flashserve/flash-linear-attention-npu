# ChunkGatedDeltaRuleBwdFinalize ATK 测试说明

本目录提供 `chunk_gated_delta_rule_bwd_finalize` 的 ATK 精度、性能、确定性和
内存检查资产。CPU 标杆完整实现 Stage 0--15 公式，不使用零输出或
NPU 结果作为标杆。

## CPU golden 接口

`scripts/chunk_gated_delta_rule_bwd_finalize_cpu.py` 提供完整 CPU 标杆：

```python
from scripts.chunk_gated_delta_rule_bwd_finalize_cpu import (
    chunk_gated_delta_rule_bwd_finalize_golden,
)

dq, dk, dv, dbeta, dg = chunk_gated_delta_rule_bwd_finalize_golden(
    q, k, v, v_new, do, du, g, beta, h, dh, a,
    q_rstd=q_rstd,
    k_rstd=k_rstd,
    beta_raw=beta_raw,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    scale=scale,
    chunk_size=64,
    use_qk_l2_norm_in_kernel=use_qk_l2_norm_in_kernel,
    use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    allow_neg_eigval=allow_neg_eigval,
    use_gate_in_kernel=False,
    state_v_first=state_v_first,
    use_exp2=True,
)
```

该函数与 `fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_bwd_finalize`
使用相同的参数顺序、默认值和五输出语义；作为独立 CPU 标杆，输入
tensor 必须位于 CPU。

## 支持范围

- Ascend 950，`q/k/v/v_new/do/du/h/dh/A` 为 BF16。
- `g/beta` 支持 BF16 或 FP32 且 dtype 必须相同，`beta_raw` 跟随 `beta.dtype`。
- `state_v_first=false/true` 分别验证 `h/dh` 末两维的 `[K,V]` 和 `[V,K]` 存储语义。
- `K=V=128`，`chunk_size=64`，支持定长和变长序列。
- `use_qk_l2_norm_in_kernel` 和 `use_beta_sigmoid_in_kernel` 为两个独立
  TilingKey 模板参数，默认值均为 `false`，支持 `false/true`。
- `allow_neg_eigval` 默认为 `false`；仅开启 beta sigmoid 时支持 `true`，
  并将 beta sigmoid backward 的系数乘 2。该开关纳入 TilingKey。
- `use_gate_in_kernel` 只支持 `false`，`use_exp2` 只支持 `true`；ATK executor
  显式传入这两个固定属性。
- `q/k` 由 ATK 生成，数据范围为 `[-0.2, 0.2]`，`v` 数据范围为
  `[-0.5, 0.5]`；`v_new/do/du/h/dh/a`
  数据范围为 `[-0.03, 0.03]`，`g/beta/beta_raw` 数据范围为
  `[-0.09, 0.09]`；CPU 节点在
  算子 ABI 边界将 ATK 升精度输入还原为实际输入 dtype，并按 NPU 各 Stage
  的 BF16 落盘和 FP32 Vector 驻留边界执行标杆。
- 输出顺序为 `dq, dk, dv, dbeta, dg`；`dbeta` 跟随 `beta.dtype`，
  `dg` 跟随 `g.dtype`。executor 会对所有输出的 shape 和 dtype 做强校验。

## 用例矩阵

| 文件 | 用例数 | 覆盖 |
|---|---:|---|
| `atk_chunk_gated_delta_rule_bwd_finalize.json` | 300 | 4 类边界 shape + 8 个模型派生 shape + 13 个泛化 shape x 12 个模板 key |
| `atk_chunk_gated_delta_rule_bwd_finalize_perf.json` | 96 | 8 个模型派生 case x 12 个模板 key |
| `atk_chunk_gated_delta_rule_bwd_finalize_mss.json` | 12 | 每个模板 key 一条尾块 GVA 用例 |

精度矩阵覆盖完整 chunk、尾 chunk、多 chunk、GVA、varlen、泛化 shape
和全部 8 个模型派生 shape。为控制 ATK CPU 标杆耗时，各派生 shape 的 `T`
约为原模型长度的四分之一；`T=64/65` 的最小完整块和尾块边界保留。

## 执行

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -scope=gen_cases
```

精度失败时必须使用 `--save_data output` 保存输出，并按 NPU 为 real、
CPU 为 expect 运行 `ct viz`。本套泛化用例按上述分组值域生成，
不得通过修改精度阈值或跳过失败用例制造通过结论。
