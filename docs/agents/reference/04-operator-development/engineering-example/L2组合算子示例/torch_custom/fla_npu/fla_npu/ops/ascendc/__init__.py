"""示例文件（形态 B/C）：torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py

交付布局依据 torch_custom/fla_npu/README.md §1.1：`_ASCENDC_OPS` 加一行 public 名即可，
短名由 `_strip_npu_prefix()` 统一导出。

注意事项：
  1. 形态 B（同一算子加 V2）**不改**这里：算子名没变，公开名与短名都不变。
  2. 形态 C（新的组合算子）加一行 `npu_op_name_fused`；它会同时导出
     `from fla_npu.ops.ascendc import op_name_fused`。
  3. 会原地写参数时在 `MUTATED_ARGUMENTS` 登记，短名与 `npu_` 前缀名都登记（照仓内既有条目）。
  4. 本示例只展示本次新增的行；实际交付时改仓库中的 `_ASCENDC_OPS`。
"""

_ASCENDC_OPS = (
    # ... 既有算子 ...
    "npu_op_name",
    "npu_op_name_fused",
)

MUTATED_ARGUMENTS = {
    # ... 既有登记 ...
}
