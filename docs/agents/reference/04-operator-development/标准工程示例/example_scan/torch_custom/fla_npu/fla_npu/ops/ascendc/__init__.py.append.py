"""示例追加片段：torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py

注意事项：
  1. `_ASCENDC_OPS` 加一行即可同时导出公开名 `npu_example_scan` 与短名 `example_scan`；
     不要在别处再维护白名单，`_get_stable_op(name)` 靠同名函数自动路由。
  2. 算子若会原地改写输入，必须在 `MUTATED_ARGUMENTS` 登记参数名；写入与否取决于参数值时，
     再登记 `MUTATION_FLAGS`（参考 npu_recurrent_kda 的 inplace_final_state）。
  3. 新算子默认没有 ctypes 回退，不需要任何声明（_LAUNCHER_ONLY_OPS 自动推导）。
  4. 导出名与算子文件、schema、适配函数、aclnn 名必须能互相推导，不允许出现只在一处使用的别名。
"""

# --- _ASCENDC_OPS 追加一行 ---------------------------------------------------
_ASCENDC_OPS = (
    # ... 既有算子 ...
    "npu_example_scan",
)

# --- 仅当算子原地写参数时追加 -----------------------------------------------
MUTATED_ARGUMENTS = {
    # ... 既有登记 ...
    # "npu_example_scan": ("initial_state",),
}
