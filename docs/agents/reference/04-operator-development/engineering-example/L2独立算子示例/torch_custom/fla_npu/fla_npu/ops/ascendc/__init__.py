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
     - "本次调用到底写不写回"取决于参数值时，再登记 `MUTATION_FLAGS`（细节见下面「三张表的分工」）；
     - 登记后：mutated tensor 的 `requires_grad=True` 会在调用前被拒绝（提示改用 functional state API），
       调用成功后自动 `increment_version()`，补齐 eager autograd 的版本检查；
     - 改 mutation 语义要跑 `tests/stable_abi/regression_mutation_contract.py`；
     - ctypes 回退路径没有 schema，原地契约只靠这份登记兜底；当前也没有 FakeTensor/functionalization，
       暂不能作为 compiler-visible op 入图。
  6. 三张表的分工（`MUTATED_ARGUMENTS` / `MUTATION_FLAGS` / `MUTATION_PREDICATES`）：
     - `MUTATED_ARGUMENTS[name] = ("state", ...)` 回答「这个算子**可能**写回哪些参数」，与本次调用的参数值无关；
     - `MUTATION_FLAGS[name] = ("开关参数名", 默认值)` 回答「**本次调用**是否真的写回」，且判据只来自**一个**参数；
     - `MUTATION_PREDICATES[name] = lambda arguments: bool` 是逃生口，只有判据需要**多个**参数组合时才用。
  7. 每次调用 wrapper 是这样判定的（对应 `_mutation_plan` / `_resolve_mutation` / `_wrap_mutable_direct_op`）：
     - 包装时一次性校验：`MUTATION_FLAGS` 里写的开关必须是该算子的真实参数，且 `bool(默认值)` 必须与
       算子签名的默认值一致；写错会在**首次调用**直接报错（`names unknown argument` /
       `default ... disagrees with ...`），不会静默判断错；
     - 调用时读开关值（位置参数、关键字参数、或省略时用默认值三种形式都支持）：
       falsy → 判定本次**不写回**，直接返回空列表，跳过 `requires_grad` 检查与 version 推进（快路径，不做
       `inspect.signature().bind(apply_defaults=True)`，那次 bind 在解码算子上约 12us/次）；
       truthy → 判定本次会写回 `MUTATED_ARGUMENTS` 里列出的张量，进入 mutation 契约；
     - 为什么必须写对：判成"没写回"会漏掉 `requires_grad` 拒绝与 `increment_version`，autograd 看到输入被改
       却没有版本推进；判成"写回"则会误拒 `requires_grad=True` 的调用、并对没改过的张量推进版本（上层会看到
       假的"被原地修改"）。
  8. 只有一个参数决定写不写回时才用 `MUTATION_FLAGS`；"调用就一定写回"（例如 conv 的 `conv_states`、
     recurrent 的 `state` 无条件更新）只需要 `MUTATED_ARGUMENTS`，不要硬塞一个开关进去。
  9. 若算子的原地契约在 `_stable.py` 的 wrapper 里自带（`_fla_npu_inplace_contract` 标记的那四个热路径），
     `_get_direct_op` 不会再二次包装，避免一次调用 bump 两次 version；两种实现方式二选一，不要都做。
"""

# ---- _ASCENDC_OPS：public 名加一行 -----------------------------------------
_ASCENDC_OPS = (
    # ... 既有算子 ...
    "npu_op_name",
)

# ---- MUTATED_ARGUMENTS：这个算子「可能」写回哪些参数（与本次调用的参数值无关）----
MUTATED_ARGUMENTS = {
    # ... 既有登记 ...
    # 示例：本算子把 initial_state 写回调用方张量时这样登记（两种拼写都写）
    # "op_name": ("initial_state",),
    # "npu_op_name": ("initial_state",),
}

# ---- MUTATION_FLAGS：本次调用「到底写不写回」由一个参数决定时登记 -------------
# 含义：`(开关参数名, 默认值)`。
#   - 开关取 truthy（或为 True 的默认值）→ 本次会写回 MUTATED_ARGUMENTS 列出的张量；
#     wrapper 会先拒绝这些张量的 requires_grad=True，调用成功后再 increment_version()。
#   - 开关取 falsy → 本次不写回（算子写内部 scratch 并把结果作为返回值），wrapper 直接跳过整个 mutation 契约。
#   - 默认值必须与 wrapper/签名的默认值一致，否则首次调用就报错（这是故意的，避免静默判断错方向）。
#   - 只在"一个参数就能决定"时用它；多参数组合用 MUTATION_PREDICATES；无条件写回就只登记 MUTATED_ARGUMENTS。
# 仓内真实例子：npu_recurrent_kda 的 inplace_final_state 默认 True——
#   True  时 kernel 直接写调用方的 initial_state（需要 grad 拒绝 + version 推进）；
#   False 时 kernel 写内部 scratch 并作为返回值，调用方张量不变（不应做任何 mutation 处理）。
MUTATION_FLAGS = {
    # ... 既有登记 ...
    # "npu_op_name": ("inplace_state", True),
}

# ---- MUTATION_PREDICATES：判据需要多个参数组合时的逃生口（会走 bind 慢路径）----
# 例：只有 "开关 A 为真且开关 B 为假" 时才写回 state：
# "npu_op_name": lambda arguments: bool(arguments["use_state"]) and not bool(arguments["write_through"]),
MUTATION_PREDICATES = {
    # ... 既有登记（当前仓内为空）...
}
