"""示例文件：torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py

交付布局依据 torch_custom/fla_npu/README.md §1.1/§1.2：

```python
# fla_npu/ops/ascendc/__init__.py
_ASCENDC_OPS = (
    ...,
    "npu_<op>",        # 加一行，公开名和短名都跟着导出
)
```

注意事项：
  1. 只加 `npu_<op>` 一行即可：公开名 `npu_<op>` 与短名 `<op>` 由 `_strip_npu_prefix()` 统一导出，
     不要再手写一份短名列表。
  2. 算子会原地写参数时，在 `MUTATED_ARGUMENTS` 登记参数名（必要时再登记 `MUTATION_FLAGS`）；
     仓内既有条目把**短名与 npu_ 前缀名都登记**（例如 `"causal_conv1d"` 与 `"npu_causal_conv1d"`），
     新增时照做。
  3. 没有 ctypes 回退不需要声明任何东西（`_LAUNCHER_ONLY_OPS` 自动推导）。
  4. 本示例只展示本次新增的行；实际交付时改仓库中的 `_ASCENDC_OPS` 与 `MUTATED_ARGUMENTS`。
"""

# ---- _ASCENDC_OPS：public 名加一行 -----------------------------------------
_ASCENDC_OPS = (
    # ... 既有算子 ...
    "npu_op_name",
)

# ---- MUTATED_ARGUMENTS：仅当算子原地写参数时登记（两种拼写都写） --------------
MUTATED_ARGUMENTS = {
    # ... 既有登记 ...
    # "op_name": ("initial_state",),
    # "npu_op_name": ("initial_state",),
}
