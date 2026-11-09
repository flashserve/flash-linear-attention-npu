# RecurrentGatedDeltaRule

循环门控 Delta Rule 推理算子：输出注意力结果，并原地更新逐 token State checkpoint。

## 支持范围

- A2/A3：使用 `op_kernel/arch22` 下的融合实现。
- A5：使用 `op_kernel/arch35` 下的四 token、三 State bank 流水实现。
- 所有架构统一拒绝非空 `gk`。
- 两个 host 接口 RecurrentGatedDeltaRule / RecurrentGatedDeltaRuleV2 均保留。
- Python 稳定入口始终路由到融合实现，四个融合开关关闭也使用同一个 kernel。

完整参数、dtype、shape、stride、可选项和异常见 [接口文档](docs/api.md)，
计算、内存与同步见 [设计文档](docs/design.md)。

## 输入与输出

Q/K/V 为 BF16；beta、g、State 支持 BF16/FP32。输出为 BF16，State 原地更新。
参数、维度和属性完整约束见上述接口文档。

## 调用

```python
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

out = recurrent_gated_delta_rule(
    query, key, value, state, beta=beta, g=g, scale=128 ** -0.5,
    actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
)
```

## 测试

ATK 工程位于 `tests/atk/recurrent_gated_delta_rule`。
`gen_arch22_cases.py` 生成 A2/A3 的 200 条精度、50 条确定性/内存和 8 条性能用例；
`gen_recurrent_gated_delta_rule.py` 生成包含 23 条 A5 V2 专项在内的 223 条用例。
使用独立 CPU 标杆、ATK mixed_tolerance_bm，性能依据 msprof kernel task 时长。
