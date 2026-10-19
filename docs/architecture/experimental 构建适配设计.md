# experimental 目录算子构建适配设计

> 基线：`zzy1205/flash-linear-attention-npu` 的 `main`（本文所有行号均以本仓为准，不是参考仓的行号）。
> 参考实现：`cann/ops-transformer` 的 `--experimental` 机制（`master@56f47a36`）。
> **本仓与参考仓在构建框架上有实质差异，照参考仓理解本仓会得出错误结论**，见 §6.2。

## 1. 一句话结论

`experimental/` 不是「另一套编译系统」，而是**一棵与稳定树结构同构、由同一个 `ENABLE_EXPERIMENTAL` 开关二选一进入构建的平行目录树**。

隔离不靠「过滤算子清单」，而靠 **`if / else` 的结构互斥**：两棵树在同一份 CMake 里走互斥分支，物理上不可能同时进入构建图。

## 2. 与既有 overlay 机制的关系

| | overlay（单算子叠加层） | experimental（本方案） |
| --- | --- | --- |
| 生效时机 | **运行期**：往已装 OPP 的 `vendors/` 里塞一份新库，靠优先级抢先 | **编译期**：决定这次构建装哪些算子 |
| 改动对象 | 已装 CANN / wheel 内嵌 OPP | 构建产物 |
| 解决什么 | 只替换**某个算子**的实现，不动其余 | 新算子还在验证期，不想进稳定包 |

两者正交、可并存；本方案不依赖 overlay，也不改变 overlay 的任何行为。

## 3. 术语

| 词 | 含义 |
| --- | --- |
| **稳定树** | `fla/ops/ascendc/<域>/<算子>/`，`ENABLE_EXPERIMENTAL=OFF` 时参与构建 |
| **实验树** | `experimental/fla/ops/ascendc/<域>/<算子>/`，`ENABLE_EXPERIMENTAL=ON` 时参与构建 |
| **域** | 算子的一级分类：本仓为 `cp` / `gdn` / `kda`（另有 `common`，只放公共头，不是算子） |
| **前缀规则** | `<op>_depends` 里的路径在实验模式下由框架统一加 `experimental/` 前缀 |
| **算子目录定位** | 框架用目录名（`NAME`）推导算子名；CI 用 `op_host` 等子目录标记定位算子目录 |

## 4. 代码结构：谁 include 谁、谁调谁

改构建逻辑前必须先看清这个骨架，否则容易改错地方（尤其是「非 open 路径」）。

```
CMakeLists.txt
│
├─ :65    include(cmake/func.cmake)              ← 无条件、最靠前
│            定义 22 个 function/macro，是纯函数库（无顶层副作用）：
│            op_add_subdirectory:66   op_add_depend_directory:154
│            add_op_to_compiled_list:145（macro）  add_ops_src_copy:367
│            add_ops_info_target:237  add_compile_cmd_target:188
│            add_bin_compile_target:474  …
│
├─ :68    if (BUILD_OPEN_PROJECT)                ← ① 不支持的 SOC：编空包后早退
│  :83        return()
│
├─ :136   if (BUILD_OPEN_PROJECT)
│  :153       include(cmake/custom_build.cmake)   ← 只在 open 分支引入
│                流程编排：发现 → 挂载 → 依赖 → aclnn → adapt py → kernel → install
│                它【不】include func.cmake，直接调用 func 里定义的函数
│  :178       return()                            ← 默认路径到此结束
│
└─ :180   else()                                  ← 非 open / RTY 路径
   :181-185   include(dependencies / variables / opbuild / rty_obj_func / intf_pub_linux)
   :186   endif()
   :289   add_subdirectory(common)                ← 仅非 open 路径可达
   :303   op_add_subdirectory(OP_LIST OP_DIR_LIST) ← 仅非 open 路径可达
```

**三条要点：**

1. **依赖是单向的**：`custom_build.cmake` → `func.cmake`。`custom_build.cmake` 里唯一的 `include()` 是 `include(CPack)`，**没有** `include(func.cmake)`。
2. **`:178 return()` 之后的一切在默认构建里都不执行**。看到 `:303` 的 `op_add_subdirectory` 不要以为它是主路径 —— 主路径的调用点是 `custom_build.cmake:251`。
3. **改 `func.cmake` 会同时影响两条路径**（它 :65 就被引入），改 `custom_build.cmake` 只影响 open 路径。本方案改的 `func.cmake:70-84` 在 `op_add_subdirectory()` 内，两条路径都调用它，所以实验/稳定分支的语义必须两边都成立。

## 5. 目录约定（本方案的核心决策）

### 5.1 实验树整体镜像稳定树的仓库相对路径

```
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/                 ← 稳定树
experimental/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/    ← 实验树（同构镜像）
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── chunk_fwd_o_def.cpp
└── op_kernel/
```

**为什么必须镜像**：`func.cmake` 的前缀规则只做一件事——在 `<op>_depends` 的值前面拼 `experimental/`。本仓 `depends` 写的是**仓根相对路径**（例如 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h`），所以：

```
experimental/ + fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h
  = experimental/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h
```

只有让实验树保持同一段相对路径，这个拼接结果才存在。若改成 `experimental/<域>/<算子>/`，全仓 `depends` 都得改写成域相对形式——那是更大的改动面。

> **代价**：多一层 `fla/ops/ascendc/`，可读性差一些。换来的是**算子源码与 CMakeLists 零改动**。

### 5.2 硬性约束

| 约束 | 原因 |
| --- | --- |
| **实验树根层不要放 `experimental/CMakeLists.txt`** | 发现逻辑是 `GLOB_RECURSE …/experimental/CMakeLists.txt`（见 §8.5），会递归匹配任意深度的同名文件；根层那份会被当成一个名叫 `experimental` 的「算子」 |
| **`experimental/<...>/common` 必须是稳定树的超集** | 实验模式下只复制实验树那份 common，稳定树独有的头不会进来 → **静默缺头**（§8.4） |
| **实验算子只能依赖实验树内已有的算子** | 前缀会把依赖指到实验树；没有就被静默跳过（§8.3） |
| 同名算子可同时存在于两棵树 | 发现逻辑用目录名推算子名，两棵树在各自模式内独立，不会冲突；但不能混放到同一棵树 |

## 6. 隔离是怎么保证的

```
                       ENABLE_EXPERIMENTAL
                        ┌──────┴──────┐
                       ON             OFF
                        │              │
   闸1 发现  GLOB experimental/**   GLOB fla/ops/ascendc/**
                        │              │
   闸2 挂载  foreach(OP_DIR) add_subdirectory(<op>/op_host)   ← 路径驱动，不判断模式
                        │              │
   闸3 依赖  depends 加 "experimental/"     depends 原样
                        │              │
   闸4 过滤  --ops=<op> 在已选集合里做减法
```

**闸 1（`cmake/func.cmake` 的 `if/else`）是唯一的根本保证**：同一个 `if` 的两个分支，不是两条独立语句，两棵树不可能同时被 glob 到。

闸 2 不判断模式——它只遍历绝对路径，树的信息已编码在 `OP_DIR` 里。

闸 4 的语义要注意：`--ops=` 是在**已发现集合**里做减法，不能把另一棵树的算子拉进来。因此 `--experimental --ops=<名字>` 要求该名字**在实验树里存在**，否则选不出任何算子（静默产出空集）。

### 6.1 四处共享件边界（不是漏，是共享）

| 边界 | 说明 | 影响 |
| --- | --- | --- |
| 根级 `common/` 恒编 | `custom_build.cmake:247` `add_subdirectory(common)` 无条件执行；`experimental/` 下**没有** common 目录 | 实验包与稳定包共用同一份根级 common，改动会同时影响两者 |
| 依赖可回退到顶层 | 写 `../xxx` 会归一化回顶层（§8.3） | 是**合法手法**，但要显式知道自己在借顶层的东西 |
| `examples/` 参与挂载 | `custom_build.cmake:301/310/313` 按 `ASCEND_OP_NAME` 把 `examples/…` 追加进 `OP_DIR_LIST` | 与模式无关；`--ops` 命中 examples 目录就会加进来 |
| `moe/3rd/*` 三个特例 | `custom_build.cmake:370/373/376` 写死**顶层**路径 | 参考仓遗留，本仓 `depends` 不含 `moe` 值，**永不触发** |

### 6.2 ⚠️ 本仓 vs 参考仓：机制差异清单

这一节最重要——参考仓的文档/注释大量提到下面这些机制，**本仓一个都没有**。照参考仓理解本仓会写错设计。

| 机制 | 参考仓 | 本仓 | 依据 |
| --- | --- | --- | --- |
| 模块聚合器分类（`add_subdirectory(attention)` / `mhc` / `mamba`） | 有，`custom_build.cmake` 里按 `ENABLE_EXPERIMENTAL` 分两支 | **没有**，所有算子统一由 glob 发现后逐算子挂载 | 本 PR 删掉了那块 `add_subdirectory(experimental/attention)` |
| `COMMON_SRC_DIR` / `ALL_COMMON_SRC_DIRS`（把 common 合并复制到 `src/common`） | 有（参考仓 `custom_build.cmake` 约 1152-1234 行） | **没有**（本仓全仓 0 处） | common 只在 `add_ops_src_copy` 里改 target 名，见 §8.4 |
| `func.cmake` 里对 `common` 的 `continue()` 排除 | 有 | **没有** | common 走同一条 `add_ops_src_copy` |
| `_OP_HOST_COMPILE_PHASE` = `DEPENDENCY` 阶段（依赖编译时跳过 CANN 版本检查） | 有 | **没有**（全仓 0 处） | 本仓不存在 `require_cann_version` |
| `--run_example` 非实验模式排除 `*/experimental/*` | 有 | **没有** | 本仓 `build.sh` 无此逻辑 |
| 算子层级 | 固定 `<域>/<算子>` 2 层 | **不固定** 2～4 层 | 见 §8.1 / §9 |

## 7. 发现与挂载：`OP_DIR_LIST` 的来源与消费

`OP_DIR_LIST` 是「本次要编的算子目录」清单，全流程围绕它转。

### 7.1 四个注入点（`cmake/custom_build.cmake`，按执行顺序）

| 序 | 行 | 注入什么 |
| --- | --- | --- |
| 1 | `:249-251` | `set(OP_DIR_LIST)` 清空后立刻 `op_add_subdirectory(OP_LIST OP_DIR_LIST)` —— **主来源**。产出在 `func.cmake` 内：`:134` 逐个 `list(APPEND _OP_DIR_LIST ${OP_DIR})`、`:138` 去重、`:140` 排序、`:142` `PARENT_SCOPE` 回传 |
| 2 | `:301` / `:310` / `:313` | `examples/<op>`、`examples/<op>`、`examples/mc2/<op>` —— 按 `ASCEND_OP_NAME` 命中才加 |
| 3 | `:318` | `list(APPEND OP_DIR_LIST ${COMPILED_OP_DIRS})` —— 走**模块聚合器路由**的算子在各自 `op_host/CMakeLists.txt` 里调 `add_op_to_compiled_list()` 登记（本仓无聚合器，实际为空） |
| 4 | `:370` / `:373` / `:376` | `moe/3rd/*` 三个写死顶层的特例（参考仓遗留，本仓不触发） |

> ⚠️ 时序陷阱：第 3 项在 `:278` 的挂载循环**之后**才 APPEND。**「进了 `OP_DIR_LIST`」不等于「被 `:278` 挂载过」**。

### 7.2 消费者（按行序）

| 行 | 用途 |
| --- | --- |
| `:260` | `OP_UT_LIST IN_LIST OP_DIR_LIST` —— `ENABLE_TEST` 时判断某 UT 目录是否属于本次构建 |
| `:278` | **逐算子挂载**：`add_subdirectory(${OP_DIR}/op_host)`，若不存在 `op_host` 则 `add_subdirectory(${OP_DIR})` |
| `:325` | `ENABLE_TEST` 时扫 `<op>/tests/CMakeLists.txt` |
| `:351` | 作为入参传给 `op_add_depend_directory(OP_LIST … OP_DIR_LIST OP_DEPEND_DIR_LIST)`（这里的 `OP_DIR_LIST` 是**形参名**，实际传入的是输出变量 `OP_DEPEND_DIR_LIST`） |
| `:558` | 收集各算子 `op_graph/*_proto.h`，更新 proto |
| `:966` | install 各算子 `op_kernel/` 下的文件 |
| `:1028` | `add_bin_compile_target(COMPUTE_UNIT … OP_INFO ${OP_DIR_LIST})` —— **kernel 二进制编译目标** |

## 8. 前缀规则（`func.cmake`）

### 8.1 规则本体与两处实现

```cmake
if (ENABLE_EXPERIMENTAL)
    set(depend_info_update "experimental/${depend_info}")
else()
    set(depend_info_update ${depend_info})
endif()
```

出现两处，处理对象不同（一个是构建图，一个是 kernel 源码），互不重叠：

| 位置 | 所在函数 | 处理对象 | 产物 |
| --- | --- | --- | --- |
| `func.cmake:160` | `op_add_depend_directory()`（`:154`） | 被依赖**算子**的目录 | `OP_DEPEND_DIR_LIST` → 补挂构建图 + 隐式扩编译范围 |
| `func.cmake:540` | `add_bin_compile_target()` 里的 depends 循环（`:538-560`） | 被依赖**算子**的 kernel 源 copy（**含 common**） | `add_ops_src_copy` target |

> ⚠️ 与参考仓的关键差异：**本仓没有 `continue()` 排除 common，也没有 `COMMON_SRC_DIR` 切换逻辑**。
> 本仓对 common 的处理在 `add_ops_src_copy()` 内部（`func.cmake:438`）：只给 **target 名**加父目录前缀（`<父目录>_<target>`）避免多域重名；
> **源目录仍由 `func.cmake:551` 的 `${CMAKE_SOURCE_DIR}/${depend_info_update}` 决定**，因此同样受 experimental 前缀影响。

### 8.2 依赖被消费时会发生三件事（`custom_build.cmake:348-378`）

```cmake
:349  op_add_depend_directory(OP_LIST ${OP_LIST}  OP_DIR_LIST OP_DEPEND_DIR_LIST)
:354  foreach (OP_DEPEND_DIR ${OP_DEPEND_DIR_LIST})
:356      if (ASCEND_OP_NAME 是 all/ALL 且 路径匹配 .*attention.*)  continue()      ← ①
:362      list(APPEND ASCEND_OP_NAME ${SUB_DIR})                                  ← ②
:363-367  if (EXISTS <dep>/op_host) add_subdirectory(<dep>/op_host) else add_subdirectory(<dep>)
:369-378  moe_inplace_index_add* / moe_masked_scatter → 追加 moe/3rd/*（参考仓遗留）  ← ③
```

| # | 行为 | 影响 |
| --- | --- | --- |
| ① | `ASCEND_OP_NAME=all` 时跳过路径含 `attention` 的依赖 | 参考仓特例；本仓域是 `cp/gdn/kda`，实际不命中 |
| ② | **把依赖算子名隐式追加进 `ASCEND_OP_NAME`** | 依赖即使不在 `--ops=` 里也会被编译。所以「`--ops` 指定了什么」≠「实际编了什么」 |
| ③ | `moe/3rd/*` 写死顶层 | 本仓不触发 |

> 本仓**没有**参考仓的 `_OP_HOST_COMPILE_PHASE = "DEPENDENCY"` 与 `require_cann_version()` 门控（全仓 0 处），所以不存在「依赖阶段跳过版本检查」这件事。

### 8.3 两条路径语义

**① `../` 会归一化回顶层**

```
experimental/ + ../fla/ops/ascendc/gdn/...  →  fla/ops/ascendc/gdn/...
```

这是 CMake 的路径语义。想「实验模式下故意复用稳定树的某个依赖」，就写 `../` 前缀。

**② 解析不到是静默跳过**

`op_add_depend_directory()` 里有**两个 `continue()`，语义不同**：

| 行 | 判据 | 含义 |
| --- | --- | --- |
| `func.cmake`（依赖解析后） | `${CMAKE_CURRENT_SOURCE_DIR}/<解析结果>/op_host/CMakeLists.txt` **或** `src/<解析结果>/CMakeLists.txt` 都不存在 | 唯一的 skip 判据（与参考仓一致，两个条件任一成立即算命中）。原来 skip 时不打印任何 message |
| `func.cmake`（非 open 分支） | 仅 `NOT BUILD_OPEN_PROJECT`：`${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/<dep>` 存在 | 非 open 路径下走 built-in 实现，跳过本仓源码。**与 experimental 无关** |

**注意**：本 PR 之后，实验模式下的「解析不到」不再是全静默 —— 若实验树里没有该依赖、但稳定树里有，`fla_resolve_depend_path()` 会**回退到稳定树并打一条 message**（每个依赖只报一次）。两个都不存在时才仍然静默 `continue()`。

### 8.4 common 在实验模式下的行为与风险

本仓没有「合并复制到 `src/common`」的阶段，而是由 `add_bin_compile_target` 里对 `<op>_depends` 的循环逐个 `add_ops_src_copy`：

| 项 | 值 | 位置 |
| --- | --- | --- |
| 源目录 | `${CMAKE_SOURCE_DIR}/${depend_info_update}` → 实验模式下即 `experimental/<...>/common` | `func.cmake:551` |
| 目标目录 | `${SRC_OUT_DIR}/${_depend_op_name}` → 多域 common 都落到同一个 `common` 目录（并集） | `func.cmake:553` |
| target 名 | `${_depend_op_name}_<soc>_src_copy`；common 会被改名为 `<父目录>_...`（`attention_common_...` / `mc2_common_...`）避免多域重名 | `func.cmake:438` |

**所以实验模式下只有实验树那份 common 会被复制。** 若实验树 common 是**分叉副本**而非稳定树的超集，稳定树独有的头不会进来 → 编译期缺头；坑在于**目录存在、复制成功、不报「路径不存在」**。

> 参考仓的同类问题可作佐证：`attention/common` 稳定 126 / 实验 97（**共享仅 47**）、`mc2/common` 242 / 7（共享 1）、`gmm/common` 79 / 1（共享 0）—— 三者都**不是镜像**。
>
> 本仓当前尚未引入 `experimental/<...>/common`；一旦引入，必须保证是稳定树的**超集**（建议加离线门禁，见 §11）。

### 8.5 一处必须知道的设计假设：`GLOB_RECURSE` 的 pattern 语义

实验分支只写了一条 pattern：

```cmake
file(GLOB_RECURSE OP_HOST_CMAKE_FILES
     "${CMAKE_CURRENT_SOURCE_DIR}/experimental/CMakeLists.txt")
```

它依赖 CMake 的行为：**`GLOB_RECURSE` 会把 pattern 拆成「目录 + 文件名」，从目录递归匹配同名文件**，因此传一个字面文件名等价于「递归找所有同名文件」。

证据：稳定分支长期使用同样写法 —— `${CMAKE_CURRENT_SOURCE_DIR}/fla/ops/ascendc/CMakeLists.txt` 这个**文件并不存在**（实际有 72 个 CMakeLists 分散在 `fla/ops/ascendc/**`），而构建正常发现 36 个算子。若 `GLOB_RECURSE` 不支持该语义，稳定构建根本编不出算子。

> 若将来升级/替换 CMake 版本后有异常，第一件事就是把这条改成显式的 `experimental/**/CMakeLists.txt` 并回归验证。

## 9. 本次修改清单

### 9.1 改了什么

| # | 文件 | 改动 | 理由 |
| --- | --- | --- | --- |
| 1 | `cmake/func.cmake:70-84` | 实验分支从「按域列白名单」改为与 `else` 分支对称的 `GLOB_RECURSE …/experimental/CMakeLists.txt` | 原白名单是参考仓的域（`ffn/gmm/moe/posembedding/…`），本仓一个都不存在；其中一行还把**算子名**当域用。改后新增实验算子 / 新增实验域都不必再改本文件 |
| 2 | `cmake/custom_build.cmake:292-299` | 删除 `add_subdirectory(experimental/attention)` 整块，换成说明注释 | 本仓无模块聚合器分类，该目录不存在，`add_subdirectory()` 指向不存在目录是 **CMake fatal error** |
| 3 | `cmake/scripts/prepare.sh`（`set_env()`） | 为 `ENABLE_EXPERIMENTAL` 补默认值 `OFF` | **可读性修正，不是 bug 修复**：空串在 CMake 里等价 OFF，行为正确，但日志与 `CMakeCache` 里看不出「实验模式关闭」。注：本文件另有 25 个变量同样无默认值，本 PR 只动本方案的关键开关 |
| 4 | `scripts/ci/parse_changed_ops.py` | 路径解析改为**树根感知 + 算子目录标记定位**：识别 `fla/ops/ascendc/` 与 `experimental/fla/ops/ascendc/` 两种根；用 `op_host / op_kernel / tests / …` 标记定位算子目录边界，末尾回退到真实存在的目录；域清单改为 `cp / gdn / kda`，非算子目录（`common / 3rd / tools / tests / internal_ops`）忽略；`_get_default_name()` 改为返回空串 | 原实现假设 `<域>/<算子>/…`（2 层）或 `experimental/<域>/<算子>/…`（3 层）；本仓是 5/6 层且**算子层级不固定**（`kda/chunk_kda_fwd` 2 层、`gdn/chunk_gdn_fwd/chunk_fwd_o` 3 层、`kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post` 4 层），按固定下标取会把 `fla` 当域、`ops` 当算子 |
| 5 | `build.sh`（`parse_changed_files()`） | 删除「无匹配算子时兜底成 `fused_infer_attention_score`」 | 该算子在**本仓不存在**（0 个文件），兜底会把编译指向空目录。改为保持为空，交由 `--PR_PKG` 分支的 `-z` 检查走「本次无可编算子」 |
| 6 | `build.sh`（`CUSTOM_OPTION` 组装） | `-DENABLE_EXPERIMENTAL` 从「只在 TRUE 时追加」改为**每次都显式下发** | 实测缺陷（A5 验证）：CMake 会把上次的 `TRUE` 留在 `CMakeCache` 里。原逻辑在本次不带 `--experimental` 时**完全不下发**该开关，缓存于是保持 `TRUE`，导致「编过一次实验包后再跑普通 `--pkg` 会继续编实验树」。改为恒下发后实测缓存能从 `TRUE` 切回 `FALSE` |
| 7 | `experimental/README.md` | 新增：目录约定 + 使用方式 | 让「实验目录怎么放、怎么编」在仓库里可见 |
| 8 | `cmake/func.cmake`（新增 `fla_resolve_depend_path()`，替换两处内联前缀） | **【G2】实验模式下依赖回退**：优先 `experimental/<dep>`；实验树里没有该算子则回退 `<dep>`（稳定树），并打一条 message（每个依赖只报一次）。两处调用点必须用同一套解析：`op_add_depend_directory()`（依赖目录补挂）与 `add_bin_compile_target()`（kernel 源 copy） | 修复前：依赖没跟着搬进实验树就被**静默丢弃**，后果不是「少编一个算子」，而是没有任何算子提供 tiling OBJECT 库 → `cust_opmaster` 链接报 `cannot find -lophost_transformer_tiling_obj`（见 §10 第 10 项实测）。两处不同步会退化成「op_host 从稳定树挂上、kernel 源从实验树拷空」 |

### 9.2 有意不修改的

| 项 | 原因 |
| --- | --- |
| 3 个 `op_host/CMakeLists.txt` 里已有的 `experimental/${dep}` 手写前缀（`gdn/chunk_gated_delta_rule_bwd`、`gdn/chunk_gated_delta_rule_fwd`、`kda/chunk_kda_fwd`） | 在 §5.1 的镜像约定下与框架前缀**结果一致**，不冲突；保留可读性。若将来改成域相对 depends，需一并删除 |
| `cmake/scripts/util/ascendc_impl_build.py` | 用相对探测解析 kernel 源（`get_kernel_source` / `_get_kernel_source`），不硬编码树根；实验树内部结构同构即无需改 |
| `cmake/config.cmake`、根 `CMakeLists.txt` | 透传与 include 均已就绪 |
| `build.sh` 的 `--experimental` 解析与透传 | `:49` 默认 FALSE、`:1162-1163` 置 TRUE、`:1203` `--PR_PKG` 分派、`:1440-1441` 转 `-DENABLE_EXPERIMENTAL=TRUE`，全部就绪 |

## 10. 使用与验证

```bash
# 稳定包（默认，行为与改造前完全一致）
bash build.sh --pkg --soc=ascend910b --ops=chunk_fwd_o

# 实验包：从 experimental/ 树里选算子（--ops 的名字必须在实验树里存在）
bash build.sh --pkg --experimental --soc=ascend910b --ops=<实验算子名>

# 实验包：全部实验算子
bash build.sh --pkg --experimental --soc=ascend910b
```

**已在两台 A5 验证服务器实测**（`ascend950` / CANN 9.2.0 / cmake 3.28.3，基线 `main@67dc99d7` + 本 PR 补丁）：

| # | 验证项 | 结果 |
| --- | --- | --- |
| 1 | `bash -n build.sh` / `prepare.sh`、`py_compile` | ✅ 通过 |
| 2 | **未打补丁**代码 + `--experimental` | ✅ 复现原缺陷：`add_subdirectory given source "experimental/attention" which is not an existing directory` |
| 3 | 打补丁 + 真实实验算子（3 层深 `kda/.../internal_ops/kda_gate_bwd_post`） | ✅ configure 成功，`OP_DIR_LIST` **只含该实验算子**（隔离生效） |
| 4 | 反向隔离：实验树根放非法 `CMakeLists.txt` + 关闭开关 | ✅ configure 成功（证明未 include） |
| 5 | 同上 + 打开开关 | ✅ 报错 `experimental/CMakeLists.txt:1 Parse error`（证明确实 include） |
| 6 | `--PR_PKG --experimental` 指向 3 层深实验算子 | ✅ 解析出 `kda_gate_bwd_post`（`rc=0`） |
| 7 | `--PR_PKG` 空清单 | ✅ `exit 200`「No custom packages to build」，**不再出现** `fused_infer_attention_score` |
| 8 | `--PR_PKG` 只改稳定树路径（实验模式） | ✅ 正确忽略（`exit 200`） |
| 9 | **有 tiling 源、无依赖**的实验算子全量编包（`kda_gate_cumsum`） | ✅ `rc=0`，产出 `build_out/fla_npu_linux-x86_64.run` |
| 10 | **有依赖但未带依赖**的实验算子编包 | ❌ 失败，根因是 §12 的 G2：依赖被静默丢弃 → 没有任何算子产生 tiling OBJECT 库 → `cust_opmaster` 链接报 `cannot find -lophost_transformer_tiling_obj` |
| 11 | **有依赖且带齐依赖闭包**的实验算子编包（第二台机器） | ✅ **闭环通过**：`rc=0`；`*_tiling.cpp` 编译 **20 次**（对照：未带依赖时为 0 次）；生成 3 个 `aic-*-ops-info.ini`；产出 `build_out/fla_npu_linux-x86_64.run`；包内 `vendors/fla_npu_transformer/` 下含该算子的 `dynamic/<op>.py`、`op_proto/inc/<op>_proto.h`、`kernel/**/<op>.json`、`ascendc/<op>/`、`libcust_opmaster_rt2.0.so`、`libcust_opapi.so`，从 `experimental/` 路径编译出 **43 个 `.o`** |
| 12 | `ENABLE_EXPERIMENTAL` 缓存残留（修 §9.1#6 前后对比） | ✅ 修后：缓存 `TRUE` 时跑普通 `--pkg` 会显式下发 `FALSE` 并切回 |
| 13 | **【G2 修复后】**实验树里**只放算子、不放任何依赖**（`kda_gate_bwd_post`），`--experimental --ops=kda_gate_bwd_post` 全量打包 | ✅ `rc=0`；回退告警出现 **1 次**且精确指向 `fla/ops/ascendc/kda/chunk_kda_bwd`；`chunk_kda_bwd_tiling.cpp` 编译 **4 次**（修复前 0 次）；缺 `tiling_obj` 库 **0 处**（修复前 1 处、直接失败）；生成 3 个 `ops-info.ini`；产出 `build_out/fla_npu_linux-x86_64.run` |
| 14 | 同上场景下的 **OFF 模式回归** | ✅ configure 成功，回退告警 **0 次**（OFF 模式解析逻辑与修改前完全一致） |

**⚠️ 修正一条此前的错误预期**：`experimental/` 为空（0 个算子）时，`--experimental` **不能**成功配置。
递归 glob 匹配空集本身不报错（这点成立），但 0 个算子会导致 opbuild 不产出 `aic-*-ops-info.ini`，
随后 `generate_transformer_adapt_py` 抛 `OpFileNotExistsError`，configure 以 `Error: ops prepare build failed` 结束。
**结论：实验树至少要有一个可编算子；空树不是受支持的状态。** 若需要「开关开着但暂时没实验算子」，应把开关关掉。

**⚠️ 往实验树复制算子的操作注意**（实测踩过）：不要先拷子算子再拷父算子 ——
`cp -r <父> <已存在的目标父目录>` 会把父目录**嵌套进自己内部**（出现 `chunk_kda_bwd/chunk_kda_bwd/...`），
产生同名算子重复，进而使 `opbuild` 的 autogen 目标互相覆盖、`ops-info.ini` 生成不出来。
正确做法：`mkdir -p <dst> && cp -a <src>/. <dst>/`，且只拷**最上层**算子（`internal_ops/*` 会随之带进来）。

**仍需有权限环境补的验证**：CI 实验 job 下的完整链路、两棵树同名算子互不冲突、common 隔离（G1，方案见 [experimental 适配现状与遗留问题.md](experimental%20适配现状与遗留问题.md) §4.1）。

## 11. 未完成 / 后续项

| 项 | 说明 |
| --- | --- |
| **CI 实验 job** | 当前 `.github/workflows/ci.yml` 没有任何 job 传 `--experimental`，这条链**从未被验证**。需新增独立 job（独立产物，不与稳定包合并） |
| **`parse_changed_ops.py` 无 CI 执行路径** | 仅在 `--PR_PKG` + experimental 时调用（`build.sh:1022`）。新增实验 job 时必须先验证它 |
| **common 镜像校验门禁** | 建议加离线断言：若存在 `experimental/<...>/common`，断言它是稳定树同名目录的超集 |
| **【G2】依赖回退（已实现，见 §9.1#8）** | 实验模式下依赖优先取实验树，实验树没有则回退稳定树并打一条 message。A5 实测：实验树只放算子、不放依赖也能 `rc=0` 打包成功（§10 第 13 项）。**语义变化**：实验算子现在可以「借用」稳定树里的依赖实现，只是会在 configure 日志里留下明确告警 |
| **common 隔离（G1）** | `cmake/config.cmake:41` 把 kernel 公共头写死在稳定树 `fla/ops/ascendc/common/`，实验算子改公共头会污染稳定包（26 个算子 / 68 处引用受影响）。**推荐「切换 + 超集门禁」**：把参考仓的「静默缺头」变成 configure 期 `FATAL_ERROR`。完整分析与代码片段见 [experimental 适配现状与遗留问题.md](experimental%20适配现状与遗留问题.md) §4.1 |
| **Python / wheel 侧** | `torch_custom/fla_npu/` 完全不感知 `--experimental`。实验算子要能被 Python 调到，需要 Stable-ABI 三件套（`csrc/src/stable_<op>.cpp` + `_stable.py` wrapper + `_ASCENDC_OPS` 一行）+ 实验 OPP 装载隔离决策 |
| **`--static`** | 与实验算子未做适配 |

## 12. 风险与坑

| 坑 | 后果 | 规避 |
| --- | --- | --- |
| 实验树根层放 `CMakeLists.txt` | 多出一个名叫 `experimental` 的「算子」 | §5.2 约定禁止 |
| 实验树 common 不是超集 | 编译期缺头，且不报「路径不存在」 | §8.4 |
| **【G2】依赖只在稳定树有** | 修复前：依赖被静默丢弃 → 该 tiling 源编译 0 次 → 没有任何算子创建 `ophost_transformer_tiling_obj` → `cust_opmaster` 链接报 `cannot find -lophost_transformer_tiling_obj`（实测同一算子稳定模式因补挂依赖而成功） | **已由 §9.1#8 修复**：`fla_resolve_depend_path()` 在实验树缺依赖时回退稳定树并打 message。实测实验树只放算子也能打包成功（§10 第 13 项）。仍建议尽量把依赖搬进实验树，避免实验包隐式依赖稳定树实现 |
| 往实验树复制算子时用 `cp -r` | 若目标父目录已存在（例如先拷了子算子），会把父目录嵌套进自己内部，产生同名算子重复，autogen 目标互相覆盖，`ops-info.ini` 生成失败，configure 报 `ops prepare build failed` | `mkdir -p <dst> && cp -a <src>/. <dst>/`；只拷最上层算子 |
| `--ops` 写了实验树里没有的名字 | 静默产出空集，构建「成功」但没编东西 | §6 闸 4；用 §10 的 `message` 自查 |
| 实验树为空 | configure 在 prepare 阶段失败（0 算子 → 无 `ops-info.ini` → adapt py 报错），**不是** glob 的问题 | 实验树至少放 1 个可编算子；不要开着开关跑空树 |
| `--pkg-type=rpm/deb` 搭配 `--experimental` | 外层守卫禁止；但 `--pkg-type=all` **会绕过**该检查（守卫条件是 `!= run && != all`） | 使用者自行避免 |
| 实验包与稳定包共用根级 `common/` | 改 common 会同时影响两者 | 改后两种包都要验 |
| 依赖算子的版本门控 | **本仓没有**参考仓的 `DEPENDENCY` 阶段门控，被依赖算子会照常走完整检查 | 不要按参考仓的预期排障 |
| `ENABLE_EXPERIMENTAL` 缓存残留 | 修之前：编过一次实验包后，普通 `--pkg` 会继续编实验树 | 已由 §9.1 第 6 项修复（恒下发该开关），实测缓存能切回 FALSE |
| 内存不足的机器上全量构建 | A5 机器 1.5 GB 内存 / 256 核，默认并行度会 OOM（实测出现 `Error 137` 与一次机器重启） | 用 `-j2`～`-j4`；`prepare.sh` 内部并行度是 `nproc*2` 且无覆盖开关 |

## 附录 A · 审计中发现但未纳入本 PR 的遗留问题

以下都是**参考仓代码带过来、与本方案无关**的残留，本次只记录不改（改动会触及稳定包行为，需要单独评估）：

| 位置 | 现象 | 影响 |
| --- | --- | --- |
| `build.sh:1450` | `if [[ $ascend_op_name != *fused_infer_attention_score* ]] && [[ != *incre_flash_attention* ]]` 时加 `-DENABLE_TILING_SINK=OFF` | 这两个算子本仓不存在，条件恒真 → **所有构建都带 `-DENABLE_TILING_SINK=OFF`**。若本仓要用 tiling sink，这里是哑点 |
| `custom_build.cmake:369-378` | 三个 `moe/3rd/*` 写死顶层路径 | 本仓 `depends` 不含 `moe` 值 → 死代码 |
| `custom_build.cmake:356-360` | `ASCEND_OP_NAME=all` 时跳过 `.*attention.*` 依赖 | 本仓无 attention 域 → 死代码 |
| `scripts/ci/parse_changed_ops.py` 的 `BlackList` | 全是参考仓的 moe/mc2 算子名 | 本仓永不命中 |
| `CONTRIBUTING.md` | 举例里写 `experimental/attention` | 文档残留，易误导 |
| `cmake/scripts/prepare.sh` | 25 个变量无默认值 | 空串多数等价 0/空，行为上通常无害；本 PR 只兜了 `ENABLE_EXPERIMENTAL` |
| `--soc` 与 kernel 落地目录 | 实测 `--soc=ascend950`（`ASCEND_COMPUTE_UNIT=ascend950`，日志 453 处提到 ascend950），但 kernel 暂存到 `build/binary/ascend910b/`，`build/binary/ascend950/` 为空 | **与本方案无关**，稳定/实验两种模式都如此；但影响「950 包是否含 950 kernel」，建议单独排查 `BINARY_COMPUTE_UNIT` 的取值来源 |
