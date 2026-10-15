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
6. **描述符**：`AclTensorView` 把 `meta.sizes`/`meta.strides`/`meta.storage_offset` 原样交给 `aclCreateTensor`，storage shape 默认是扁平的 `numel`（`logical_storage=true` 且张量连续时改用逻辑 shape，见 `acl_meta.h` 的注释）。**适配层不判定布局能力，也不做 dense 拷贝**：非连续输入如实按 view 交出去，能不能正确寻址是算子的责任。OPP 侧 `isview=0` 时 storage shape 不可观测；`isview=1` 的算子（如 `RecurrentGatedDeltaRule`，手写 `op_host/op_api` 里显式 `CreateView`）会读 stride，`npu_recurrent_gated_delta_rule` 的 state 能走 stride 就靠这个。`CausalConv1d` 的 aclnn 接口是构建期生成的（没有 `op_host/op_api`），它理解 `convStates` stride 的能力跟着 CANN 走；在支持该接口之前的 CANN 上算子按连续处理并自行给 warning，适配层不干涉（见 `stable-abi-host-cost.md` 的对应一节）。

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
  调 `get_ws`/`launch`、自己拆栈：它们要把描述符作为一个 bundle 交队列
  （`detail::enqueue_launch`），宏里没有这个位置。**与参数所有权无关**——
  `Tensor(a!)` 走类型化拆栈是安全的，而且证据就是这两个适配器自己：它们现在正是
  用 `to<Tensor>` 拆 `Tensor(a!) state` / `initial_state`，FRESH 归零、服务级 32/32。
  （全库带 `Tensor(a!)` 的只有这两个 op；conv1d 的 `conv_state` 是普通 `Tensor`，
  所以它不能拿来当反例。）
  历史教训：两个适配器最初用 `to<AtenTensorHandle>` 读必选槽——那是 catch-all
  的 memcpy，**既不消费也不释放**，于是每次调用、每个新鲜输入都漏一份引用。
  910B3 实测：decode 形状、每次新建 q/k/v 的 2000 次调用让 caching allocator 涨
  191 MiB（≈100 KiB/次），conc32 服务涨到 8.2 GiB 后 OOM；改成一槽一个
  owning `Tensor` 后 FRESH 组归零、常驻 allocated 从 103.1 MiB 降到 4.6 MiB。
  可选槽仍用 `to<std::optional<Tensor>>`：它自己消费内层 handle 并 `delete` 掉
  dispatcher 分配的那个 box；换成 `to<Tensor>` 会把 box 指针当成张量 handle
  包起来、再当张量删除，直接破坏堆。
  代价与现状：
  * 它们自己复制了 workspace/launch 那一段（各约 50 行）；
  * `op_abi_parity.py` 需要知道 `AtenTensorHandle` 与 `Tensor` 等价，并允许
    适配函数末尾带 `Tensor*`/`bool*` 这类内部输出参数；
  * `op_abi_validate.py` 原先只看 `FLA_STABLE_EXEC`，看不到它们的 aclnn 参数表——
    现已补上（`hand_written_calls`），并额外做适配↔ctypes 表的一对一对拍。
  收敛方向：把"描述符 bundle 交队列"这件事做进宏（`boxed.h` 多一个入口），
  两个适配器即可整体转成宏写法；转换前必须先过 parity + 多线程多 stream +
  vLLM 单请求。
- **conv1d_update 的 `conv_state` 没写 `Tensor(a!)`**：它和两个 recurrent 入口一样是
  in/out ref（schema 只写成 `Tensor conv_state`），契约靠 Python 侧
  `MUTATED_ARGUMENTS` 兜底（拒绝 requires_grad + 手动 bump version counter），eager
  路径行为正确。补上标注也不会改变结果——单独编一个只改 schema 的库，conv1d 组
  28/28 与基准逐位一致——但它**不是**编译模式的修复：recurrent 的
  `state`/`initial_state` 同样是 in/out ref 且不在返回值里，功能化执行下 mutation
  要靠返回值承载。今天 vLLM-Ascend 的服务路径是 eager（`--enforce-eager` 关掉了
  torch.compile 与 CUDAGraph），所以这条不构成本次发布风险；要支持图模式时，把
  state 放进返回值才是完整改法，两个 recurrent 入口一起改。
- **`solve_tri` 的 `tnd`**：该 OPP 上 kernel 直接杀进程（ctypes/launcher 都一样），薄层包装里显式拒绝，避免把非法输入变成崩溃。
- **conv1d FN + `has_initial_state`**：初态序列的输出行在 kernel 里不可复现（同一 ctypes 调用两次结果差 260，第三次是 0），回归里按 kernel 级记录并只对 `has_initial_state=False` 的区间断言 parity。
- **`int[]` 只能是 host int32/int64 tensor**：device tensor 会被 `int_values` 拒绝（否则按 host 指针读 device 内存）。
- **`char*` 只能取表内取值**：非法字符串在 Python 侧报错、非法 code 在 C++ 侧报错，与 ctypes"把任意字符串交给 kernel"不同型。
- **pybind 后端已删除**（`csrc_thin/`、`_thin.py`、`_C_thin` 及其测试、打包开关）。原计划的删除前置条件是 launcher 通过真实 vLLM 服务验证（plan 的 R20）；本次按要求先跳过该验证直接移除，vLLM 服务级回归需在删除后补跑。
