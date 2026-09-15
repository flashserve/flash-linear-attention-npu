# Stable-ABI 薄层设计（手写适配 + 共享宏）

本文描述当前实现的结构、约定和门禁。面向两类读者：想知道"一次调用怎么走"的人，和要新增算子的人（后者直接看 [stable-abi-op-onboarding.md](stable-abi-op-onboarding.md)）。

## 1. 目标与约束

- **无 ABI 依赖**：不依赖 CPython ABI（不是扩展模块），不依赖 torch_npu 的 C++ ABI（只用 `torch/csrc/stable` 的 `aoti_torch_*` C 面）。因此同一个 `.so` 可以在不同 torch/torch_npu/Python 之间复用。
- **host 开销与 vLLM 同量级**：调用路径上不建 Python 对象（无逐次 `ctypes` 描述符、无 `getattr` 链、无 `inspect.signature`）。
- **可读、可审计**：一个算子 = 一个 C++ 适配 + 一个 Python 包装，参数顺序在三处（schema、适配函数、aclnn）一致，并由离线门禁检查。

## 2. 层次

```
调用方 (fla / vLLM)
  └─ fla_npu.ops.ascendc._stable.<op>()     真签名的 Python 包装：名字映射 + int[]/枚举转换
       └─ torch.ops.fla_npu_stable.<op>(...)  STABLE_TORCH_LIBRARY 注册的算子（boxed）
            └─ boxed_adapter<run_<op>>      按 schema 拆栈，调用适配函数，打包返回
                 └─ FLA_STABLE_EXEC(...)     RAII 持有 aclTensor/aclIntArray/字符串/标量
                      └─ aclnn<Op>GetWorkspaceSize → workspace → aclnn<Op>
```

文件：

| 文件 | 职责 |
| --- | --- |
| `fla_npu/ops/ascendc/_stable.py` | 每算子一个包装；`_host_ints`（int[]）、`_char_code`/`_ENUM`（枚举）、`_current_stream_ptr` |
| `csrc/src/stable_ops.cpp` | 单 TU：包含各适配文件，`m.def`/`m.impl` 注册，构建戳符号 |
| `csrc/src/stable_<family>.cpp` | 适配实现：`kSchema_<op>` + `run_<op>` |
| `csrc/include/stable/boxed.h` | `boxed_adapter`：按函数签名拆栈/打包 |
| `csrc/include/stable/exec.h` | RAII 参数持有者 + `FLA_STABLE_EXEC` 宏 |
| `csrc/include/stable/acl_meta.h` | `TensorMeta`、`AclTensorView`、`AclIntArrayView`、分配/元数据读取 |
| `csrc/include/stable/at_facade.h` | 最小 ATen 形状门面（dtype 常量、`TensorOptions`） |
| `csrc/include/stable/layout_math.h` | 各 layout 下的 token/head/dim/chunk 数 |

## 3. 书写约定（门禁依赖这些）

1. **参数顺序三处一致**：`kSchema_<op>` 的形参顺序 = `run_<op>` 的形参顺序 = `FLA_STABLE_EXEC` 转发给 aclnn 的顺序（schema 里仅用于控制输出的布尔参数不转发，例如 `output_final_state`）。由 `tools/op_abi_parity.py` 检查前两者，`tools/op_abi_validate.py` 检查 aclnn 那一侧（对 OPP 头文件）。
2. **`int[]` 跨边界用 host int64 tensor**：schema 里写 `Tensor?`，Python 侧 `_host_ints(...)`，C++ 侧 `int_array(...)` 读 host 指针并 `aclCreateIntArray`。device 元数据（vLLM 的 `query_start_loc` 等）走 `optional_tensor(...)`，两者语义不同但 schema 类型相同，区别在调用点。
3. **`char*` 跨边界用 int code**：schema 里写 `int`，Python 侧 `_char_code(op, arg, value)` 查 `_stable._ENUM`，C++ 侧 `cstr(k<Op><Arg>Names, code)` 查名表并做边界检查。**名表顺序必须与 `_ENUM` 一致**，由 `stable_coverage.py` 检查。所有 layout 参数统一用 `BSND=0, BNSD=1, TND=2, NTD=3`（`layout_math.h` 的 `layout::Code`）。
4. **可选输出**：schema 写 `Tensor?`，缺席时 C++ 返回 `std::nullopt`；`boxed.h` 的 `pack` 会把它写成 **boxed optional**——直接塞 tensor handle 会让 dispatcher 当指针解引用（`chunk_fwd_h` 段错误就是这么来的）。
5. **原地修改**：在 `__init__.py` 的 `MUTATED_ARGUMENTS` 登记参数名；写入与否取决于参数值时再登记 `MUTATION_FLAGS`（例如 `npu_recurrent_kda` 的 `inplace_final_state`）。
6. **描述符**：`AclTensorView` 对连续张量用逻辑 shape 作 storage shape（与 `nd_tensor` 一致），非连续退化为 `(numel,)`。OPP 侧 `isview=0` 时该字段不可观测；`isview=1` 的算子（如 `RecurrentGatedDeltaRule`）必须与参考一致——这是 `npu_recurrent_gated_delta_rule` 的 state 走 stride 的关键。`isview` 不是算子属性而是 runtime 行为：同一份 OPP 下 CANN 9.1.0 的 `CausalConv1d` 拿不到 `convStates` 的 stride、9.2.0 拿得到（见 `stable-abi-host-cost.md` 的 CANN 版本一节），所以 `_dense_conv_state` 只对拿不到的那一档保留 dense staging。

## 4. 构建戳

`csrc/build_stable.py` 把 `csrc/{src,include}`（含文件名、CRLF 归一化）哈希后：

- 编入 `.so`：`-DFLA_STABLE_SOURCE_HASH=...`，导出 `fla_npu_stable_source_hash()`；
- 写入 `fla_npu/ops/ascendc/_stable_hash.py::SOURCE_HASH`。

`_stable.load()` 比对两者，不一致直接报错并给出重编命令。这样"换了适配但没重编 .so"不会静默按旧 stack index 发 kernel。

## 5. 门禁

全部离线（不需要 NPU）：

| 工具 | 检查 | 失败含义 |
| --- | --- | --- |
| `tools/stable_coverage.py` | 每个公开 `npu_*` 都有 wrapper / schema / `run_` / 注册；C++ 枚举名表与 `_stable._ENUM` 一致 | 有算子没接上，或枚举顺序漂移 |
| `tools/op_abi_parity.py` | schema 形参 vs 适配函数形参（名字与类型） | 位置型拆栈会静默错位 |
| `tools/op_api_parity.py` | `_stable` 对 ctypes 的公开签名（参数名、顺序、默认值） | 调用方换后端会 TypeError |
| `tools/stable_ctypes_fallbacks.py` | 没有适配层回退到 ctypes | 回退会带上描述符森林的开销 |
| `tools/op_abi_validate.py` | ctypes 参数表与每个 `FLA_STABLE_EXEC` 对 **OPP 头文件** | aclnn 形参变了（例如 `stateVFirst`）而调用点没跟 |
| `tools/op_abi_validate.py`（同一次运行） | 调用点与 ctypes 表**互相对拍**（不需要 OPP） | 两条实现路径的参数顺序/类型漂移 |
| `tools/op_abi_validate.py --json` | 同上，产出报告 | — |

需要 NPU 的验证都在 `tests/stable_abi/`（运行方式见那里的 README）：
`regression_stable_full.py` 是主驱动（逐算子 ctypes↔launcher 逐位对比 + 场景集基线，
`--group` 可只跑一组），`regression_ops.py` 是场景库、`regression_950_ops.py` 是
Ascend950 专属场景，`test_stable_stream_interleaving.py` 覆盖多线程多 stream，
`regression_mutation_contract.py` 覆盖原地更新与 autograd 契约，
`customer_switch_compat.py` 验证客户可见面（签名/返回值/backend）没有变化，
`bench_stable_host.py` 是单算子 host A/B。

Ascend950（A5）的现状：`--group a5` 在 950 上跑 `regression_950_ops.py` 的四个场景
（`fwd_prepare` / `bwd_finalize` / `chunk_gated_delta_rule_fwd` 的 A5 路径 /
`chunk_gated_delta_rule_bwd`），其余场景由 950 全量矩阵覆盖；两条基线
（Ascend910B3 / Ascend950PR_9579）都在 `stable_abi/stable_scenarios.json` 里，
`FLA_NPU_BASELINE_WRITE=1` 重录。

## 6. 已知边界（记录，不隐藏）

- **recurrent 家族仍是 pre-macro 写法**：`npu_recurrent_gated_delta_rule` 与
  `npu_recurrent_kda`（`csrc/src/stable_recurrent_gdr.cpp` /
  `stable_recurrent_kda.cpp`）不经过 `FLA_STABLE_EXEC` + `boxed_adapter`，而是自己
  调 `get_ws`/`launch`、自己拆栈。原因是宏的拆栈会**夺走参数 handle 的所有权**：
  torch 的 `torch::stable::Tensor(AtenTensorHandle)` 把 handle 包进带删除器的
  `shared_ptr`（`tensor_struct.h:80-83`），`to<Tensor>` 明确写明
  "steals ownership of the input's underlying AtenTensorHandle"
  （`stableivalue_conversions.h:622-624`）。recurrent 的 `state`/`initial_state` 正是
  `Tensor(a!)` 这种被调用方别名的参数，所以这两个适配器保留了 handle 形式。
  代价与现状：
  * 它们自己复制了 workspace/launch 那一段（各约 50 行）；
  * `op_abi_parity.py` 需要知道 `AtenTensorHandle` 与 `Tensor` 等价，并允许
    适配函数末尾带 `Tensor*`/`bool*` 这类内部输出参数；
  * `op_abi_validate.py` 原先只看 `FLA_STABLE_EXEC`，看不到它们的 aclnn 参数表——
    现已补上（`hand_written_calls`），并额外做适配↔ctypes 表的一对一对拍。
  收敛方向（待设备验证）：让 `boxed.h` 支持"以借用的 `AtenTensorHandle` 拆栈"的
  第二种拆栈方式，或在设备上证明类型化拆栈对 `Tensor(a!)` 安全后整体转换。
  两个方案都必须先过 parity + vLLM 单请求（现任实现是唯一在真实服务里验证过的）。
- **`solve_tri` 的 `tnd`**：该 OPP 上 kernel 直接杀进程（ctypes/launcher 都一样），薄层包装里显式拒绝，避免把非法输入变成崩溃。
- **conv1d FN + `has_initial_state`**：初态序列的输出行在 kernel 里不可复现（同一 ctypes 调用两次结果差 260，第三次是 0），回归里按 kernel 级记录并只对 `has_initial_state=False` 的区间断言 parity。
- **`int[]` 只能是 host int32/int64 tensor**：device tensor 会被 `int_values` 拒绝（否则按 host 指针读 device 内存）。
- **`char*` 只能取表内取值**：非法字符串在 Python 侧报错、非法 code 在 C++ 侧报错，与 ctypes"把任意字符串交给 kernel"不同型。
- **pybind 后端已删除**（`csrc_thin/`、`_thin.py`、`_C_thin` 及其测试、打包开关）。原计划的删除前置条件是 launcher 通过真实 vLLM 服务验证（plan 的 R20）；本次按要求先跳过该验证直接移除，vLLM 服务级回归需在删除后补跑。
