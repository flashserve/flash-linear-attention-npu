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
  5. 原地契约（登记后由 `_wrap_mutable_direct_op` 统一执行，不要自己再包一层）：
     - 被改写的参数必须在 `MUTATED_ARGUMENTS` 登记，短名与 `npu_` 前缀名都登记；
     - "是否写回"取决于参数值时再登记 `MUTATION_FLAGS`，格式 `(参数名, 默认值)`，例如
       `"npu_recurrent_kda": ("inplace_final_state", True)`；只有多参数判据才用 `MUTATION_PREDICATES`；
     - 登记后：mutated tensor 的 `requires_grad=True` 会在调用前被拒绝（提示改用 functional state API），
       调用成功后自动 `increment_version()`，补齐 eager autograd 的版本检查；
     - 改 mutation 语义要跑 `tests/stable_abi/regression_mutation_contract.py`；
     - ctypes 回退路径没有 schema，原地契约只靠这份登记兜底；当前也没有 FakeTensor/functionalization，
       暂不能作为 compiler-visible op 入图。
"""

# ---- _ASCENDC_OPS：public 名加一行 -----------------------------------------
_ASCENDC_OPS = (
    # ... 既有算子 ...
    "npu_op_name",
)

# ---- MUTATED_ARGUMENTS：仅当算子原地写参数时登记（两种拼写都写） --------------
MUTATED_ARGUMENTS = {
    # ... 既有登记 ...
    # 示例：本算子把 initial_state 写回调用方张量时这样登记（两种拼写都写）
    # "op_name": ("initial_state",),
    # "npu_op_name": ("initial_state",),
}

# ---- MUTATION_FLAGS：仅当"是否写回"取决于参数值时登记 ------------------------
MUTATION_FLAGS = {
    # ... 既有登记 ...
    # 格式 (参数名, 默认值)；默认值必须与 wrapper 的默认值一致
    # "npu_op_name": ("inplace_state", True),
}
