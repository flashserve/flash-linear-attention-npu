# 算子工程结构规范

> 规范角色：`ENGINEERING-STRUCTURE`
>
> 规范版本：`V1`
>
> 适用范围：`fla/ops/ascendc/**` 下算子与调用层的新增、迭代和检视。只规定"工程怎么摆"：
> 目录、文件、各层契约、命名、模板参数、接口迭代和编码细节。
>
> 不规定内容：数学公式与 Stage 划分见 [`03-方案设计.md`](../../03-方案设计.md)；逐 Stage 实现顺序见
> [`04-算子开发.md`](../../04-算子开发.md) 与 [`chunk-dependent-development.md`](chunk-dependent-development.md)、
> [`chunk-independent-development.md`](chunk-independent-development.md)；用例来源与验收见
> [`tests/atk/README.md`](../../../../tests/atk/README.md)。
>
> 参考实现（本文所有规则的落地样本）：
>
> | 参考算子 | 覆盖的规范点 |
> | --- | --- |
> | [`chunk_kda_fwd_prepare`](../../../../fla/ops/ascendc/kda/chunk_kda_fwd_prepare/) | def 输出全 `REQUIRED`；可选输出只在 L2 表达；`outputMask` 档位拼接；arch22/arch35 双实现；host tiling UT |
> | [`chunk_kda_fwd`](../../../../fla/ops/ascendc/kda/chunk_kda_fwd/) | aclnn V1→V2 迭代；私有 L0 融合 + 组合入口；四段 launch；模板与 tiling key 双 key |
> | [`chunk_fwd_h`](../../../../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/) | arch22/arch35 同名实现；`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` 模板展开 |
> | [`chunk_bwd_dv_local`](../../../../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/) | 最小完整形态：host + kernel + aclnn + example + tests |
>
> 本文与 [`开发者指南.md`](../../../开发者指南.md)（场景 3）互补：开发者指南给"新增算子要碰哪些文件"，
> 本文给"每个文件必须长成什么样、哪些做法会被检视拦下"。

## 1. 分层与职责

一次算子交付沿四层落地，每层只回答自己那一层的问题，不允许跨层代答：

| 层 | 目录 | 回答的问题 | 不允许做的事 |
| --- | --- | --- | --- |
| L0 kernel + host tiling | `fla/ops/ascendc/<模块>/<算子>/` | 数学怎么在 AiCore 上分核、分 Stage 执行 | 解释 autograd 重计算策略；按调用方决定要不要算 |
| L2 aclnn | `op_host/op_api/aclnn_<op>.{h,cpp}` | 公开 C 接口：张量契约、可选输出、场景拦截 | 决定 Python 侧默认值；读调用方框架状态 |
| Stable-ABI 适配 | `torch_custom/fla_npu/csrc/src/stable_<op>.cpp` | 把 torch 参数拆栈并下发 aclnn | 判定布局能力、补 dense 拷贝、缓存 stream |
| Python 公开接口 | `torch_custom/fla_npu/fla_npu/ops/ascendc/` | 入参名/默认值/返回 tuple 的 FLA 语义 | import `torch_npu` 作为默认路径、注册 `torch.ops.npu` |

层间契约的详细设计见 [`适配层设计.md`](../../../architecture/适配层设计.md)，新增适配的逐步操作见
[`适配层接入指南.md`](../../../architecture/适配层接入指南.md)。

## 2. 标准目录结构

以 `chunk_kda_fwd_prepare` 为样本的完整形态：

```text
fla/ops/ascendc/<模块>/<算子>/
|-- CMakeLists.txt                        # 遍历子目录，按 ENABLE_TEST/BENCHMARK 决定是否进 tests
|-- README.md                             # 算子能力、输入限制、输出布局（唯一定义来源）
|-- docs/
|   |-- api.md                            # 全部公开接口、返回码、可选输出语义
|   `-- design.md                         # 方案详设，开头记录规则版本
|-- op_host/
|   |-- CMakeLists.txt                    # add_op_to_compiled_list / add_modules_sources / add_ops_compile_options
|   |-- <算子>_def.cpp                    # OpDef：Input/Output/Attr/AICore config
|   |-- <算子>_infershape.cpp             # 可选：shape 推导（多数算子由 tiling 侧承担）
|   |-- <算子>_tiling.cpp / .h            # tiling 入口
|   |-- <算子>_tiling_processor.h         # tiling 计算主体
|   |-- <算子>_output_mask.h              # 有多档可选输出时必须：档位/掩码常量
|   |-- arch22/<算子>_tiling_impl.h       # 平台专用 tiling 实现（A2/A3，按需）
|   |-- arch35/<算子>_tiling_impl.h       # 平台专用 tiling 实现（A5，按需）
|   |-- op_api/
|   |   |-- <算子>.h / .cpp               # L0：内部 exec（opdev / l0op）
|   |   |-- aclnn_<算子>.h / .cpp         # L2：公开 aclnn 接口（V1）
|   |   `-- aclnn_<算子>_v2.h             # V2 迭代入口的公开声明（实现见 §5.2）
|   `-- tests/                            # host/tiling 单测（ENABLE_TEST 时编译）
|       |-- CMakeLists.txt
|       `-- <算子>_tiling_processor_test.cpp
|-- op_kernel/
|   |-- <算子>.cpp                        # 唯一 kernel 入口：模板参数 + 架构选择 + tiling key 分派
|   |-- <算子>_struct.h                   # TilingData 与常量
|   |-- <算子>_tiling_key.h               # 必须：ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_SEL
|   |-- <算子>_<stage>.h                  # 按 Stage 拆分的实现头
|   |-- arch22/                           # A2/A3 专用实现（按需），文件名与根目录同名
|   `-- arch35/                           # A5/ascend950 专用实现，文件名与根目录同名
|-- examples/                             # 可选：test_aclnn_<算子>_*.cpp 直调示例
`-- tests/                                # 可选：算子自带脚本/ATK 资产
```

上表是**必需集合**：`_tiling_key.h` 属于必需件（见 §4.2），清单里没有再列 `_policy.h`、`_common.h`
这类"可有可无"的文件——需要公共工具时按实际内容建同名文件，但不要为了对齐目录树而空建。

对应的调用层文件（与算子目录一一对应，缺一不可）：

```text
torch_custom/fla_npu/
|-- csrc/src/stable_<算子>.cpp           # kSchema_<算子> + run_<算子>（一算子一文件）
|-- csrc/src/stable_ops.cpp              # #include 一行 + m.def/m.impl 两行
|-- fla_npu/ops/ascendc/_stable.py       # 真签名 Python wrapper
`-- fla_npu/ops/ascendc/__init__.py      # _ASCENDC_OPS 公开名；原地参数登记 MUTATED_ARGUMENTS

tests/atk/<算子>/
|-- README.md                            # 输入限制、标杆来源、TilingKey 覆盖表、验收结果
|-- atk_<算子>.json / _perf.json / _mss.json
|-- <算子>.yaml
|-- gen_<算子>.py
`-- executor_<算子>.py
```

## 3. host 层规范

### 3.1 `op_host/CMakeLists.txt`

**算子发现规则**（`cmake/func.cmake` 的 `op_add_subdirectory`）：构建系统用
`GLOB_RECURSE fla/ops/ascendc/CMakeLists.txt` 找算子目录，算子名 = `op_host` 的上一级目录名，
并要求 `op_host/CMakeLists.txt` 里调用过 `add_op_to_compiled_list()`。由此得出三条硬约束：

1. 任何形态（含只有 L2 的组合算子）都**必须**有 `op_host/CMakeLists.txt`；漏掉时整个目录不进构建，
   而且不会有任何报错。
2. 打开 `ENABLE_TEST` 时，构建系统要求 `${OP_DIR}/tests/CMakeLists.txt` 存在，否则**跳过该算子**；
   不做单测的算子要在 README 里写明，避免 CI 静默跳过。
3. 目录里有 `op_kernel_aicpu/` 而未开 `ENABLE_AICPU` 时，该算子同样被跳过。

固定由四段组成，新增算子按相邻算子抄写，不要自创结构：

```cmake
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE <算子>_def.cpp)
endif()
add_modules_sources(OPTYPE <算子> ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME <OpName>
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if(ENABLE_TEST)
    add_subdirectory(tests)
endif()
```

规则：

1. 保持 `--cce-auto-sync=off`，不得改为 `on`；同步由代码显式表达。
2. 平台相关编译选项（如 `ascend950` 的 `COMPUTE_UNIT Ascend950PR_9599`）只加在平台分支里。
3. 算子间依赖（复用其他算子的 kernel 源）用 `set(<算子>_depends "...;...")` 声明，并按依赖闭包
   展开 `add_subdirectory`：过滤构建（`FLA_NPU_OPS` / `--ops` 只列主算子）时，依赖算子的 `op_host`
   必须一起进本次构建；打包侧由 `cmake/custom_build.cmake` 按同一份 `${<算子>_depends}` 把依赖算子的
   `op_kernel` 产物安装到 impl 目录。完整的展开循环见
   [`engineering-example/L2组合算子示例/.../op_name_fused/op_host/CMakeLists.txt`](engineering-example/L2组合算子示例/fla/ops/ascendc/ops_classify/op_name_fused/op_host/CMakeLists.txt)。
4. 依赖缺失的症状是**运行期**报 `aclnnStatus=561103` 且 `Config_Error(EZ1013): ... the JSON
   configuration file of operator ... cannot be found` / `AclOpKernelInit failed`，编译与安装阶段都成功；
   这是依赖配置缺失，不是算子数值或 kernel 缺陷。**全量构建通过不能证明过滤构建自洽**，改动组合入口后
   必须用一次"只列主算子"的过滤构建 + 一条走组合路径的端到端用例验证。真实案例：`chunk_kda_fwd` 的
   V2 组合入口缺少 `chunk_fwd_h` 的配置（Issue #695，由 #705 系列合入修复）。
5. 只有 L2 的组合算子（没有 def）不写 `target_sources(op_host_aclnnExc ...)`，其余结构相同
   （见 [`engineering-example/L2组合算子示例/`](engineering-example/L2组合算子示例/)）。

### 3.2 `def`：输入、输出与属性

`*_def.cpp` 是 ABI 敏感文件（见 [`repository-rules.md`](../../../repository-rules.md) 的 ABI 兼容性），
改 `Input`/`Output`/`Attr` 的类型、数量、必选/可选属性都会被 CODEOWNERS 拦下重检。

**输出规则（硬性）**：

1. `def` 中的 `Output` **一律写 `ParamType(REQUIRED)`**，不写 `ParamType(OPTIONAL)`。
2. 调用方"可以不传某个输出"属于 **L2 语义**：在 `aclnn_*.h` 里以可空描述符
   （`const aclTensor *xxxOut`，`nullptr` 表示本次不导出）表达，由 L2 推导档位并拼接。
3. 不通过把输出标成 `OPTIONAL` 来表达"这个结果可以不产出"。理由：
   - kernel 是否写该槽位由 tiling 档位决定，不由 def 的 required/optional 决定，标 `OPTIONAL` 会让 def 与
     L2 语义出现两份互相矛盾的事实；
   - `def` 属于 ABI 敏感路径（[`repository-rules.md`](../../../repository-rules.md)），把"调用方可选"写进 def 后，
     像 V2 那样"只新增一个可选输出"的迭代也必须改 def，从而触发 ABI 重检；
   - 新增输出时只要在 def 里加 `REQUIRED` 一行、在 L2 里加一个可空指针，改动范围可预测。
4. 反例与现状：`chunk_kda_fwd`、`chunk_fwd_h`、`causal_conv1d_bwd`、`chunk_kda_bwd` 等较早算子的 def
   仍保留 `ParamType(OPTIONAL)` 输出，属于历史形态。迭代既有算子时**不得再新增**；是否需要把历史输出
   改成 `REQUIRED` 单独按 ABI 变更评估，不能顺手改。
5. 正确做法见 `chunk_kda_fwd_prepare`：def 的 13 个输出全部 `REQUIRED`，L2 侧
   `qHatOut/kHatOut/qRstdOut/kRstdOut/betaEffOut` 用可空指针表达，并按非空组合推导
   `none/forward/recompute/save` 四档。

**输入规则**：

1. 可选输入写 `ParamType(OPTIONAL)`；由值决定行为的元数据（`cu_seqlens`、`chunk_indices`）同时写
   `.ValueDepend(OPTIONAL)`。
2. dtype/format 列表按"模板实例顺序"排列，同一算子的所有输入输出列表长度必须一致。
3. 不存在"预留但永远不读"的输入；确实预留时必须在 `README.md` 与 `docs/api.md` 写明当前语义和拦截条件。

**属性规则**：

1. 有默认值的属性写 `AttrType(OPTIONAL)` + 默认值；无默认值的写 `AttrType(REQUIRED)`。
2. 属性是公开语义，改名/改类型等价于接口变更，必须同步 `docs/api.md`、Python 入口和测试。
3. 只用于 L2 内部档位推导的属性（例如把档位编码进 op 的 `output_mode`）必须在 `docs/api.md` 说明它不是用户参数。

**AICore config**：`ascend910b`、`ascend910_93`、`ascend950` 三个 config 用同一份 `OpAICoreConfig`，
平台差异放 tiling/模板，不复制算子定义。

### 3.3 tiling

1. tiling 入口只做三件事：校验、计算 TilingData、`SetTilingKey`。校验结论必须能直接对应 `README.md` 的支持范围。
2. 每个 workspace 区域记录 size、offset、使用方和生命周期；同一块 workspace 被多个 Stage 复用时写明复用条件。
3. TilingData 字段与 kernel 侧 `<算子>_struct.h` 一一对应，字段名一致，不允许 kernel 侧隐式假设未写入的字段。
4. Tail、partial chunk、varlen 的有效长度必须由 tiling 计算并传入，kernel 不自行反推。
5. TilingKey 用 `GET_TPL_TILING_KEY(...)` 生成，实参顺序与 `op_kernel/<算子>_tiling_key.h`
   的 `ASCENDC_TPL_ARGS_DECL` 一致；workspace 总量必须计入 `platform.GetLibApiWorkSpaceSize()`；
   平台判定用 `platform.GetCurNpuArch()`（A5 = `NpuArch::DAV_3510`）。
6. `<算子>_tiling_processor.h` 的切分/offset/workspace 计算写成可独立调用的函数或类，并**在头文件内实现**：
   host UT 只 include 这个头就能覆盖分支，不需要造完整 `TilingContext`，也不需要额外链接目标。
7. 空 tensor、非法 `cu_seqlens`/`chunk_indices`、超出位宽的 varlen 元数据都在这里拦截并打印实际值
   （对应 [#577](https://github.com/flashserve/flash-linear-attention-npu/issues/577)、
   [#508](https://github.com/flashserve/flash-linear-attention-npu/issues/508)）。

### 3.4 `op_api`：L0 与 L2 的分工

| 文件 | 层 | 职责 |
| --- | --- | --- |
| `<算子>.h/.cpp` | L0 | 内部 exec：按已解析好的张量做校验、分配输出、下发 kernel；可被其他 L0 组合调用 |
| `aclnn_<算子>.h/.cpp` | L2 | 公开 C 接口：`GetWorkspaceSize` / `Launch` 成对；参数校验、可选输出拼接、场景拦截、返回码 |
| `aclnn_<算子>_v2.h` | L2 | V1 冻结后的迭代入口声明；实现放同一算子的 `aclnn_<算子>.cpp` 尾部或单独的 `aclnn_<算子>_v2.cpp`，见 §5.2 |

规则：

1. L2 不接收、也不解释 autograd 重计算策略；`output_final_state`/`disable_recompute`/`return_intermediate_states`
   这类保留策略只存在于 Python 与 legacy 包装层。
2. L2 只做张量与算法契约校验；布局能力判定、dense 拷贝不做（非连续输入如实按 view 交给算子）。
3. L2 的报错必须写明触发条件与实际值（例如 `outputMask=0x%x`、实际 `Kdim/Vdim`），不能只写"参数错误"。
4. 返回码只允许 `ACLNN_ERR_PARAM_INVALID`（用户参数）、`ACLNN_ERR_INNER_NULLPTR`/`ACLNN_ERR_INNER_*`（内部失败）
   两类，并与 `docs/api.md` 的"返回值"章节逐条对齐。
5. V1 与 V2 共用同一份参数校验与输出指针语义；重复代码只允许出现在参数表本身，语义必须集中在一处。

### 3.5 host 侧平台分支

1. 平台专用 tiling 实现放 `op_host/arch22/*.h`（A2/A3）与 `op_host/arch35/*.h`（A5），
   入口按 SoC 选择；`arch22` 与 `arch35` 都允许有专用文件，不要求只有 A5 才分平台。
2. 同一平台的 tiling 实现不止一份 `.cpp` 时，放 `op_host/op_tiling/arch35/*.cpp`；
   `cmake/obj_func.cmake` 会把该目录收进 tiling 目标。目录名不能自创。
   `fla/` 内当前实际使用 `arch22`、`arch32`、`arch35`，而 cmake 的收集与排除逻辑还会识别
   `arch20`、`arch31`、`arch38`（见 `cmake/ut.cmake`、`cmake/scripts/get_soc_version.py`）。
3. 平台差异不得表现为"复制一整套 host 代码"；公共校验、TilingData 结构与任务描述保持一份。

### 3.6 `op_host/tests`

1. 位置：`op_host/tests/<算子>_*_test.cpp` + `CMakeLists.txt`，仅在 `ENABLE_TEST` 打开时编译。
2. 用途：验证 tiling 的分支选择、`outputMask` 档位推导、任务切分和边界条件（不替代 ATK 精度验收）。
3. 覆盖要求：每个可达 TilingKey 至少一条用例，输出掩码/档位组合用 `static_assert` + 单测双重保护
   （参考 `chunk_kda_fwd_prepare_output_mask.h` 与 `op_host/tests/chunk_kda_fwd_prepare_tiling_processor_test.cpp`）。

## 4. kernel 层规范

### 4.1 根目录与 `arch22`/`arch35`

1. 只允许两种布局，按"两个平台的实现差异有多大"选一种：
   - **默认 + A5 差异**：根目录放默认实现与全部公共头，`arch35/` 只放 A5 差异文件，文件名与根目录同名
     （参考 `chunk_bwd_dv_local`）。
   - **A2/A3 与 A5 各一份**：根目录只放公共头，`arch22/` 与 `arch35/` 各放一份同名平台实现
     （参考 `chunk_fwd_h`、`chunk_kda_fwd_prepare`）。`arch22` 不是"没有专用文件才用根目录"，
     它同样可以是专门的平台实现目录。
2. 同一份实现不允许出现第三份副本：能共用就提到根目录，不能共用才按平台拆。
3. 架构选择的唯一依据是编译期宏，写在算子入口 `.cpp` 顶部：

   ```cpp
   #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
   #include "arch35/<算子>_cube.h"
   #include "arch35/<算子>_vec.h"
   #else
   #include "arch22/<算子>_cube.h"
   #include "arch22/<算子>_vec.h"
   #endif
   ```

   参考 `chunk_fwd_h/op_kernel/chunk_fwd_h.cpp` 与 `chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp`。
4. 差异只放在被拆分的实现头里；两份内容相同的文件（TilingData 结构、策略头等）保留在根目录一份，
   不要复制到 `arch22/`、`arch35/` 各一份。
5. 不新增第三个架构目录名；新的平台集合变化先改 `cmake/scripts/get_soc_version.py` 的目录排除表和本节。
6. 平台实现里最容易出问题的是同步：buffer 复用前缺少反向同步（WAR hazard）、尾块或特定 head 配置下漏事件
   导致 hang，参考 [PR #700](https://github.com/flashserve/flash-linear-attention-npu/pull/700)、
   [#325](https://github.com/flashserve/flash-linear-attention-npu/issues/325)、
   [#462](https://github.com/flashserve/flash-linear-attention-npu/issues/462)。

### 4.2 TilingKey 与模板参数

模板化 TilingKey 是**必需件**：每个 Ascend C 算子都必须有 `op_kernel/<算子>_tiling_key.h`，
用 `ASCENDC_TPL_ARGS_DECL` 声明模板参数、`ASCENDC_TPL_SEL` 枚举实例，kernel 入口按模板参数实例化。
不允许用"单实例 + 运行期分支"代替模板化；`op_host` 的 tiling 侧通过
`#include "../op_kernel/<算子>_tiling_key.h"` 复用同一份枚举，`TilingKey 与选择条件一一对应`
（见 [`04-算子开发.md`](../../04-算子开发.md) §3）。

1. TilingKey 只表示**场景族**，不表示平台、不表示独立算子、不表示接口：
   - `chunk_kda_fwd`：`key=2` 表示 `chunk=64 && K=V=128`（含 dense/tail/varlen），`key=1` 表示其余通用场景；
     A2/A3/A5 都生成同样的两个 key，平台在 key 内部由编译架构选择。
2. 编译期模板参数只给"必须在该实例中消除的分支"开口，典型为 dtype 档、`V_DIM`/`K_DIM` 档、以及
   `USE_GK`/`USE_EXP2`/`STATE_FP32`/`STATE_V_FIRST` 这类布尔开关。
3. 模板声明与实例枚举写在 `<算子>_tiling_key.h`，用 `ASCENDC_TPL_ARGS_DECL` 声明、`ASCENDC_TPL_SEL` 枚举。
   `ASCENDC_TPL_*` 与 TilingData 不混放：`_tiling_key.h` 只放模板参数与实例，`_struct.h` 只放
   TilingData 与常量（部分历史算子把两者合并在 `_struct.h`，新算子不沿用）；
   笛卡尔积用宏逐层展开（见 `chunk_fwd_h_tiling_key.h` 的 `_TPL_SEL_GATE/_EXP/_STATE/_LAYOUT/_ONE`），
   不手抄组合。
4. 模板实例内部用 `if constexpr` 消除不需要的分支；禁止用运行期 `if` 判断编译期常量。
5. 同一 key 内运行期分支（dense/tail/varlen、是否带初始状态）仍要有各自的精度用例与 `_mss.json` 用例。
6. 不支持组合必须在 host 拦截并返回 `ACLNN_ERR_PARAM_INVALID`，不得"模板没有实例所以静默走别的分支"。
7. 修改 TilingKey、模板选择条件或分派分支时，必须同步更新算子 ATK README 的 TilingKey 覆盖表并给出实际选择证据
   （[`tests/atk/README.md`](../../../../tests/atk/README.md)「TilingKey 覆盖交付」）。

### 4.3 入口分派

1. 一个算子只有一个 kernel 入口 `<算子>.cpp`，即使内部包含多个阶段（Prepare/Post-WU/FwdH/Finalize）。
2. 需要"分阶段多次 launch"时，用私有属性（例如 `stage`）在同一入口内分派，不新增 L0 原型、不新增公开接口
   （参考 `chunk_kda_fwd` 的 `KDA_STAGE_*`）。
3. 入口只做地址解析与分派，具体 Stage 实现放在 `<算子>_<stage>.h`。
4. 每个 Stage 的写回、workspace slot 释放、同步与 §3.3 的 tiling 描述一致。

## 5. 调用层与接口迭代

### 5.1 可选输出在 L2 拼接

1. def 的输出全部 `REQUIRED`（§3.2）；L2 用可空描述符 `const aclTensor *xxxOut` 表达可选输出，
   并在头文件注释里写清 `nullptr` 的语义。必选输出由 L2 显式校验非空并返回
   `ACLNN_ERR_PARAM_INVALID`，不靠"形参能不能传空"来区分。
2. L2 由"哪些输出指针非空"推导执行档位/掩码，并把结果传给 L0 与 tiling；档位组合必须显式枚举，
   非法组合返回 `ACLNN_ERR_PARAM_INVALID`（参考 `GetOutputMask` 与 `PREPARE_*_OUTPUT_MASK`）。
3. 输出缺席时分两种处理，必须在设计文档中写明是哪种：
   - **内部必需但对外可选**（例如 Finalize 要用的 `hCompute`）：L2/kernel 仍按必需张量分配或使用 workspace 承接，
     只是不公开；
   - **纯增量导出**（例如反向需要的 `qHat/kHat/qRstd/kRstd/betaEff`）：缺席时按当前档位要求分配内部张量或直接跳过，
     不影响计算结果。
4. 调用方传了张量时，优先 `ReuseOrAlloc` 直接复用为内部目标；需要改变布局/顺序时由 L2 做一次明确的
   `ViewCopy`（例如公开 `hOut` 由 head-major 转 sequence-major），并把这个拷贝计入性能说明。
5. **同一输入下，传与不传可选输出必须计算结果逐位一致**，只有是否落公开 GM 的区别；该等价性要有对应用例。
   新增保存值（`q_hat/k_hat/q_rstd/k_rstd/beta_eff` 一类）就是这样做的：`def` 不动、只在 L2 表达，
   参考 [#694](https://github.com/flashserve/flash-linear-attention-npu/issues/694)。

### 5.2 aclnn V1 / V2 迭代规范

当已发布入口需要新增能力（新开关、新可选输出、新场景）时，按以下规则迭代：

1. **V1 冻结**：已发布入口的函数名、形参数量与顺序、形参类型、返回码、公开输出布局不得修改。
2. **V2 新开**：新能力写成 `aclnn<Op>V2GetWorkspaceSize` / `aclnn<Op>V2`，其余层（L0、kernel、def）不动。
   声明固定放 `op_host/op_api/aclnn_<op>_v2.h`；实现可放在已有的 `aclnn_<op>.cpp` 尾部（V1 的公开头
   此时完全不改，参考 `chunk_kda_fwd`：`aclnn_chunk_kda_fwd_v2.h` + 实现在 `aclnn_chunk_kda_fwd.cpp`），
   也可单独放 `aclnn_<op>_v2.cpp`。两种都可以，但不能改 V1 头。
3. **共用契约**：V1 与 V2 共用参数校验、输出指针语义、返回码契约和公开输出布局；同一输入下公开输出一致。
4. **差异集中**：V1/V2 的差别只来自文档化开关与场景范围。V2 头文件顶部用注释列出：与 V1 的关系、
   支持范围、每个新增开关的语义和默认值。
5. **新增可选输出只加在 V2 形参尾部**，def 不新增输出（§3.2）；缺席时语义与 §5.1 一致。
6. **默认值等于历史语义**：V2 新增开关的默认值必须复现 V1 行为（例如 `epsilon=1e-6`、
   `useQkL2normInKernel=false`、`useBetaSigmoidInKernel=false`、`useExp2=true`）。
7. **不得静默忽略开关**：开关被显式打开但当前场景不在 V2 支持范围内时，Python 入口直接报错并提示场景要求，
   不允许静默降级到 V1 或忽略该开关。
8. **场景选择在适配层 `run_`**：V1/V2 是同一个 schema 的两条 aclnn 入口，由
   `torch_custom/fla_npu/csrc/src/stable_<op>.cpp` 的 `run_<op>` 内部按参数/形状分支选择
   （参考 `stable_chunk_kda_fwd.cpp` 在 `aclnnChunkKdaFwdV2` 与 `aclnnChunkKdaFwd` 之间的分支）；
   Python wrapper 只做参数校验、默认值补齐与透传，调用方也只面对一套 Python 签名。
9. **交付同步**：`docs/api.md` 必须同时记录两个入口的签名、关系表、场景差异与返回码；`docs/design.md` 记录
   组合关系；ATK 用例覆盖 V1、V2 与回退路径；`tools/op_abi_validate.py` 能对 OPP 头文件校验所有 `FLA_STABLE_EXEC` 调用点。
10. **ABI 重检**：只新增 V2 文件、不动 def 和 V1 aclnn 时，不触发额外 ABI 检视；一旦改动 def 或既有 aclnn 的
    入参类型/返回值/必选可选属性，按 [`repository-rules.md`](../../../repository-rules.md) 请求 owner 检视。

### 5.3 fla_npu 调用接口兼容性规范

1. **入口唯一**：Ascend C 算子的主入口是 `fla_npu.ops.ascendc.<op>`；短名 `<op>` 由 `__init__.py` 的
   `_ASCENDC_OPS` 一并导出。不新增其他主入口，`torch.ops.npu` 只是可选的 legacy 兼容路径。
2. **只追加关键字参数**：新增参数一律加在参数表末尾并带默认值；默认值等于本算子的历史行为，已有调用结果不变。
3. **返回 tuple 稳定**：返回值的个数、顺序、语义固定；新增输出只追加在末尾，未请求时返回 `None`（schema 的 `Tensor?`）。
4. **保留策略留在这层**：`output_final_state`/`disable_recompute`/`return_intermediate_states` 等由 Python 与 legacy
   包装层解释成"传哪些输出指针"，不下沉到 L2。
5. **三处参数顺序一致**：`kSchema_<op>` 形参 = `run_<op>` 形参 = `FLA_STABLE_EXEC` 下发的 aclnn 实参顺序；
   由 `tools/op_abi_parity.py` 与 `tools/op_abi_validate.py` 离线门禁检查。
6. **跨边界类型**：枚举/layout 走"名表 + int code"（名表顺序与 `_stable._ENUM` 一致）；`int[]` 只走 host int64 CPU 张量；
   返回的 `Tensor?` 槽必须走 boxed optional，不能塞裸 handle。
7. **stream 每次现取**：`_current_stream_ptr()` 在每次调用时读取，禁止缓存为进程级变量。
8. **原地参数登记**：会改写输入的算子，在 `MUTATED_ARGUMENTS`（必要时 `MUTATION_FLAGS`）登记；
   契约回归由 `tests/stable_abi/` 覆盖。
9. **一算子一文件**：新建 `csrc/src/stable_<op>.cpp`，在 `stable_ops.cpp` 加 `#include` 一行与注册两行；
   不新增手写入口（两个 recurrent 适配是 pre-macro 历史遗留，不作为模板）。
   完整交付件只有四个文件，以 [`torch_custom/fla_npu/README.md`](../../../../torch_custom/fla_npu/README.md) §1.1 为准：
   `csrc/src/stable_<op>.cpp`（新建）、`csrc/src/stable_ops.cpp`（include 一行 + `m.def`/`m.impl` 两行）、
   `fla_npu/ops/ascendc/_stable.py`（真签名 wrapper，不要 `*args`/`**kwargs`）、
   `fla_npu/ops/ascendc/__init__.py`（`_ASCENDC_OPS` 加一行 public 名）。不要自造 `*.append.*`
   之类的中间文件名，直接改仓库中的同名文件。
10. **不需要 ctypes**：新算子默认没有 ctypes 回退，`_LAUNCHER_ONLY_OPS` 自动推导；不得为了"兼容"回退到 ctypes。
11. **门禁**：`tools/stable_coverage.py`、`tools/op_abi_parity.py`、`tools/stable_ctypes_fallbacks.py`、
    `tests/test_stable_gates.py` 全部通过后才算接口交付完整。

### 5.4 组合入口与"只有 L2 接口"的算子

公开接口不一定都要配一套新的 kernel。按实现形态分三类，选择依据是"这套计算能否由已有算子拼出来"：

| 形态 | 目录特征 | 参考 |
| --- | --- | --- |
| A 独立实现 | `def` + `op_kernel` + L0 + L2 齐全 | 大多数算子，如 `chunk_kda_fwd_prepare` |
| B 自研主体 + L0 组合 | 本算子有自己的 `def`/kernel，同时另有一个组合入口，在 L0 层拼接其它算子 | `chunk_kda_fwd` 的 V2（Prepare → FwdH → Finalize） |
| C 只有 L2 | 没有新的 `def` 与 `op_kernel`，L2 直接调用其它算子的 L0 并把结果拼成公开输出 | 组合型入口 |

规则：

1. **C 形态不新增 `def`**：没有新 kernel 就没有新 op 原型；L2 通过 `#include` 依赖算子的
   `op_host/op_api/<依赖算子>.h`，直接调用其 L0 实现。
2. B/C 两种形态的 L2 只做四件事：入参校验、layout 与连续化处理、按组合顺序调用 L0、把内部张量
   拼接成公开输出（`ReuseOrAlloc` 复用调用方张量，必要时一次 `ViewCopy`）。
3. 组合入口必须复用被组合算子的 L0，不得复制其 kernel，也不得在 L2 里重写其数学。
4. 依赖关系写进 `op_host/CMakeLists.txt`：`set(<算子>_depends "...")` 覆盖 L0 头与 kernel 源，并按
   依赖闭包展开构建（构建侧）——`cmake/custom_build.cmake` 会按同一份变量打包依赖产物（打包侧）。
   只声明不展开时，过滤构建产出的包不自洽：跑到组合入口才以 `561103` + `EZ1013`（缺 JSON 配置）失败，
   容易被误判为数值问题；详见 §3.1 第 3–4 条与 Issue #695。
5. 组合入口要显式声明支持范围（哪些 dtype/layout/shape 走组合、其余回落哪个入口），不满足时返回
   `ACLNN_ERR_PARAM_INVALID`；支持范围与回落规则写进 `docs/api.md`。
6. C 形态仍必须交付：schema 与公开名、Stable-ABI 适配（`csrc/src/stable_<算子>.cpp`）、`docs/api.md`、
   ATK 用例；被组合算子各自的 TilingKey 覆盖表由它们自己维护，组合入口只记录"调用了哪些 key"。

## 6. 测试看护

三层看护各自负责不同粒度，不能互相替代：

| 层 | 位置 | 负责 |
| --- | --- | --- |
| 仓级门禁 | `ci/run_checks.sh`、`tests/test_stable_gates.py`、`torch_custom/fla_npu/tools/*` | 构建、wheel 与 OPP 布局、ABI 与适配层一致性、公开名导出 |
| 单算子精度/性能看护 | `tests/atk/<算子>/` | 精度、性能、确定性、mssanitizer、TilingKey 覆盖、验收结果 |
| 算子内 UT/示例 | `op_host/tests/`、`examples/test_aclnn_<算子>_*.cpp` | tiling 分支、`outputMask` 档位、aclnn 直调 |

### 6.1 `tests/atk/<算子>/` 交付要求

1. 必备文件：`README.md`、`atk_<算子>.json`、`atk_<算子>_perf.json`、`atk_<算子>_mss.json`、`<算子>.yaml`、
   `gen_<算子>.py`、`executor_<算子>.py`，可选 `scripts/`。
2. 三份 JSON 来源不同、不能互相替代：精度用例来自生成器筛选补充；性能用例来自用户模型 case；
   `_mss.json` 按全部可达 TilingKey 人工构造。
3. 算子 ATK README 必须建立三类映射：逻辑分支 → 精度 case id、模型 case → 性能 case id、
   可达 TilingKey → `_mss.json` case id，并给出实际选择证据。
4. 验收结果写入算子 ATK README：标杆与版本、目标 SoC、用例总数/失败数、覆盖结论、回归结论；
   存在性能目标时逐模型 case 列表比较（不能用平均值掩盖未达标 case）。
5. 用例的 shape、dtype、layout 必须同时满足算子 README、tiling 校验和 executor 输入构造。
6. 不得提交 `atk_output/`、`result/`、xlsx、profiling/sanitizer 日志、`__pycache__` 等产物。

### 6.2 新增或修改算子时的看护动作

1. 新增算子：先落 `README.md` + `docs/api.md` + `docs/design.md`，再落 `tests/atk/<算子>/` 骨架，
   最后补精度、性能与 `_mss.json` 用例。
2. 修改 TilingKey / 模板 / 分派分支：更新覆盖表 + 补 `_mss.json` 用例 + 给出实际选择证据。
3. 修改 def / aclnn 签名 / schema / `op_plugin` 适配：按 ABI 敏感路径请求 owner 检视，并重跑适配层门禁。
4. 修改公共组件或 runtime：列出全部受影响算子，逐个跑单算子看护，并补充 affected-ops 的 Example/ST。
5. 新增或修改组合入口（形态 B/C）或 `<算子>_depends`：补一次"只列主算子"的过滤构建，并用一条走组合
   路径的端到端用例确认包自洽（全量构建不会暴露缺依赖配置的问题）。
6. 涉及分核、归约顺序或同步的改动：补确定性回归（重复运行 + 逐位比较），参考
   [#440](https://github.com/flashserve/flash-linear-attention-npu/issues/440)；引用 ATK runner 的结论前先确认
   scope 判定本身正确，参考 [#365](https://github.com/flashserve/flash-linear-attention-npu/issues/365)。

## 7. 编码细节规范

1. **命名**：目录与文件名 `snake_case`，与算子名一致；`OP_ADD` 用 PascalCase（`ChunkKdaFwdPrepare`）。
   固定后缀：`_def.cpp`、`_infershape.cpp`、`_tiling.cpp/.h`、`_tiling_processor.h`、`_struct.h`、`_policy.h`、
   `_common.h`、`_tiling_key.h`、`_output_mask.h`。不要自创后缀。
2. **文件头**：新增源文件保留仓库文件头许可声明（跟随同目录相邻文件的 BSD 3-Clause 或 CANN 2.0 版本）；
   头文件用 include guard（全大写下划线）。
3. **参数校验集中在 host**：校验函数只判断、不计算；空 tensor、空序列、非法 `cu_seqlens`、不支持的
   dtype/layout/shape 组合都在这里拦截并返回明确错误信息。
4. **报错文本**：写出实际触发的条件与实参值（`Kdim/Vdim`、`layout`、`outputMask`、实际 `chunk_size`），
   并指明当前支持的取值；不写"参数错误"这类无上下文文本。
5. **同步**：跨 pipe/核同步只用当前 CANN 头文件支持的成对 `SetFlag`/`WaitFlag`、`CrossCoreSetFlag`/
   `CrossCoreWaitFlag` 组合；生产者复用 slot 前必须有反向 flag/credit；`PipeBarrier` 只处理同 pipe 依赖。
   详细前置知识见根 [`AGENTS.md`](../../../../AGENTS.md)「算子开发前置知识」。
6. **资源与生命周期**：UB/L1 队列按 `TPipe`/`TQue` 语义管理；手工 buffer 必须显式记录
   free/being-written/ready/being-read 状态；`FreeTensor` 后不得再访问。
7. **常量集中**：输出掩码、flag id、slot 数、档位枚举用命名常量或 `struct`/`enum` 集中管理，
   并配 `static_assert` 校验互斥与全覆盖；禁止散落魔法数字。
8. **临时/生成物**：不提交 `build/`、`build_out/`、`dist/`、`output`、`PROF_*`、`atk_output/`、
   `__pycache__`、临时脚本和调试输出。提交前执行 `git status --short` 与 `git diff --check`。
9. **格式**：按仓根 `.clang-format` 格式化 C/C++；Python 侧跟随同目录风格。
10. **文档同步**：能力与输入限制写算子 `README.md`；接口与返回码写 `docs/api.md`；方案与规则版本写
    `docs/design.md`。三份文档与代码同时更新，不允许只改代码。

## 8. 交付前校验清单

新增或迭代算子时逐条确认：

- [ ] 目录树与 §2 一致；没有自创后缀或未登记的子目录。
- [ ] `op_host/CMakeLists.txt` 四段齐全，`--cce-auto-sync=off` 未被改动。
- [ ] `def` 的输出全部 `REQUIRED`；没有新增 `ParamType(OPTIONAL)` 输出。
- [ ] 可选输出的缺席语义只在 L2 表达，且档位/掩码组合有显式枚举与非法组合拦截。
- [ ] L1 与 L2 的职责边界清楚：L2 不解释重计算策略，不做 dense 拷贝，报错带实际值。
- [ ] 根目录 kernel 是默认（arch22）实现，`arch35/` 只放差异文件，架构选择用 `__CCE_AICORE__ == 310`。
- [ ] TilingKey 只标场景族；模板参数用 `ASCENDC_TPL_*` 声明与枚举，内部用 `if constexpr`。
- [ ] 组合入口的 `<算子>_depends` 已按依赖闭包展开（构建侧 + 打包侧各一份），并用"只列主算子"的
      过滤构建验证过包自洽（不出现 `561103` / `EZ1013`）。
- [ ] `kSchema_`/`run_`/`FLA_STABLE_EXEC` 三处参数顺序一致，公开名已加入 `_ASCENDC_OPS`。
- [ ] 新增关键字参数在末尾且默认值等于历史行为；返回 tuple 只追加。
- [ ] V1 未被修改；V2 的差异、支持范围与默认值在 `aclnn_*_v2.h` 与 `docs/api.md` 写清。
- [ ] `tests/atk/<算子>/` 三份 JSON、README、yaml、gen、executor 齐全，TilingKey 覆盖表有实际选择证据。
- [ ] SoC 覆盖（A2/A3/A5）与 varlen/tail/边界用例齐备。
- [ ] 文档（README/api/design）与代码、报错、返回码一致。
- [ ] `git status --short`、`git diff --check` 干净，无生成物与敏感信息。

## 9. 历史 issue 索引（写本节规范时依据的坑）

下表是撰写/修订本规范时扫过的仓库 issue，按"规范条目 → issue"组织。新增或迭代算子时，先看与本任务相关的行；
标 `OPEN` 的行表示问题仍未收敛，按当前规范实现时要额外确认。

| 主题 | 相关条目 | Issue / PR | 状态 | 需要注意什么 |
| --- | --- | --- | --- | --- |
| 过滤构建缺依赖闭包 | §3.1、§5.4 | [#695](https://github.com/flashserve/flash-linear-attention-npu/issues/695)、[#618](https://github.com/flashserve/flash-linear-attention-npu/issues/618) | #695 已修复 / #618 未收敛 | 症状是运行期 `561103` + `EZ1013`（缺 JSON 配置），编译与安装都成功；`<算子>_depends` 必须在构建侧展开、打包侧安装，且要用"只列主算子"的过滤构建验证 |
| 构建参数缺少前置校验 | §3.1、§7.3 | [#482](https://github.com/flashserve/flash-linear-attention-npu/issues/482) | OPEN | `FLA_NPU_OPS` 指定不存在的算子时应尽早报错并列出候选，而不是编译到一半或静默通过 |
| 内嵌 OPP 初始化顺序 | §5.3、§6 | [#429](https://github.com/flashserve/flash-linear-attention-npu/issues/429) | 已修复 | `CANN` 先于 `fla_npu` 初始化时内嵌 OPP 未注册，同样报 `561103`；这类"看起来像算子错"的问题要先查运行时/安装状态 |
| 空 tensor 与异常场景拦截、报错文本 | §3.3、§7.3、§7.4 | [#577](https://github.com/flashserve/flash-linear-attention-npu/issues/577)、[#561](https://github.com/flashserve/flash-linear-attention-npu/issues/561)、[#558](https://github.com/flashserve/flash-linear-attention-npu/issues/558)、[#641](https://github.com/flashserve/flash-linear-attention-npu/issues/641) | #577 已修复 / 其余 OPEN | 空 tensor、非法元数据必须在 host 拦截并打印实际值与原因，不能只给错误码；"异常场景未拦截"是同类问题的重复出现 |
| varlen 元数据的位宽与计数 | §3.3、§7.3 | [#508](https://github.com/flashserve/flash-linear-attention-npu/issues/508) | OPEN | 总 token 数超过 `65536` 时 tiling 失败，属于 count/offset 位宽与上界检查问题；tiling 侧所有乘法都要做溢出检查 |
| 第三方框架传入的 rank/shape 语义 | §3.2、§7.3 | [#615](https://github.com/flashserve/flash-linear-attention-npu/issues/615) | OPEN | MindSpore 侧 `OriginalShape` 展平会导致 rank 误判；`def`/tiling 的 rank 判据要写清依赖哪一维，不能假设调用方总是传标准 layout |
| 计算顺序/缩放顺序 | §4.2、§7 | [#563](https://github.com/flashserve/flash-linear-attention-npu/issues/563) | OPEN | 与参考实现不同的"先加后乘"会改变精度；改变计算顺序要同步更新 design 与 golden 语义 |
| 同步与事件时序（含 UB hazard、hang） | §4.1、§7.5 | [PR #700](https://github.com/flashserve/flash-linear-attention-npu/pull/700)、[#325](https://github.com/flashserve/flash-linear-attention-npu/issues/325)、[#462](https://github.com/flashserve/flash-linear-attention-npu/issues/462) | 均已修复 | 典型形态是尾块/特定 head 配置下的缺反向同步、WAR hazard 或 hang；buffer 复用前必须有反向同步或 free 计数 |
| 输出 bitwise 确定性 | §4.2、§6 | [#440](https://github.com/flashserve/flash-linear-attention-npu/issues/440) | 已修复 | varlen 路径曾出现逐位不确定；分核/归约顺序变化要跑确定性 scope |
| 内存检测与内存占用看护 | §6.1 | [#575](https://github.com/flashserve/flash-linear-attention-npu/issues/575)、[#614](https://github.com/flashserve/flash-linear-attention-npu/issues/614) | 已修复 | `_mss.json` 与内存检测必须真正命中 sanitizer 版本包；内存占用类结论按用例逐条记录 |
| ATK 结果误判 | §6、§6.1 | [#365](https://github.com/flashserve/flash-linear-attention-npu/issues/365)、[#428](https://github.com/flashserve/flash-linear-attention-npu/issues/428) | 已修复 | runner 的 scope 调用与结果判定本身出过错；不要只看 shell 退出码，要看总任务数/失败数/精度结论 |
| 精度失败与复检口径 | §6.1、检视 skill | [#731](https://github.com/flashserve/flash-linear-attention-npu/issues/731)、[#519](https://github.com/flashserve/flash-linear-attention-npu/issues/519)、[#529](https://github.com/flashserve/flash-linear-attention-npu/issues/529)、[#543](https://github.com/flashserve/flash-linear-attention-npu/issues/543)、[#554](https://github.com/flashserve/flash-linear-attention-npu/issues/554)、[#534](https://github.com/flashserve/flash-linear-attention-npu/issues/534)、[#640](https://github.com/flashserve/flash-linear-attention-npu/issues/640) | 多数 OPEN | 单轮 `FAIL` 不等于算子错；先区分数值误差/无效区/标杆语义，再做 `accuracy_lt` + `ct dual analyze` 复检；禁止用收窄 range、删 case、放宽阈值制造通过 |
| 可选输出与新增入口的迭代方式 | §3.2、§5.1、§5.2 | [#694](https://github.com/flashserve/flash-linear-attention-npu/issues/694)、[#696](https://github.com/flashserve/flash-linear-attention-npu/pull/696)、[#702](https://github.com/flashserve/flash-linear-attention-npu/pull/702) | 已合入 | 反向保存值（`q_hat/k_hat/q_rstd/k_rstd/beta_eff`）就是这样加的：`def` 不动，只在 L2 用可空描述符表达，V1 签名保持不变 |
| 非连续输入与全局状态副作用 | §5.3 | [#491](https://github.com/flashserve/flash-linear-attention-npu/issues/491)、[#636](https://github.com/flashserve/flash-linear-attention-npu/issues/636) | OPEN | 非连续 state 会走 stride 路径，注意 host enqueue 与服务性能；import/初始化不得留下影响未调用算子的全局状态 |
| 公开文档/注释脱敏 | §7.10 | [#714](https://github.com/flashserve/flash-linear-attention-npu/pull/714) | 已修复 | README/注释里不能出现本地路径、临时目录、日志路径等本地调测信息 |
| 本规范的来源 | 全文 | [#182](https://github.com/flashserve/flash-linear-attention-npu/issues/182)、[#221](https://github.com/flashserve/flash-linear-attention-npu/issues/221)、[#410](https://github.com/flashserve/flash-linear-attention-npu/issues/410)、[#298](https://github.com/flashserve/flash-linear-attention-npu/issues/298) | #182/#221/#410 已闭环 / #298 OPEN | docs/agents 的五个阶段与参考资料分层由这些需求演进而来；工程结构规范是阶段 4 的实现侧补充 |

维护要求：

1. 新增规范条目时，如果它来自某个 issue/PR，在同一行补链接；只写编号，不暴露内部信息。
2. issue 状态变化（尤其 `OPEN` 行收敛）后更新"状态"列，避免用过期结论指导实现。
3. 本表只收录能映射到具体规范条目的问题；纯性能优化、一次性环境问题不进表。
