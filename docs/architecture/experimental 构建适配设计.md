# experimental 目录算子构建适配设计

> 状态：**构建侧改造已实现**（本 PR）；CI 侧与 Python 侧见 §9「未完成 / 后续项」。
> 适用于 `fla/ops/ascendc/` 之外的「实验算子」——即尚未稳定、随时可能增删的算子。
> 参考实现：`cann/ops-transformer` 的 `--experimental` 机制（`master@56f47a36`）。

## 1. 一句话结论

`experimental/` 不是「另一套编译系统」，而是**一棵与稳定树结构同构、由同一个 `ENABLE_EXPERIMENTAL` 开关二选一进入构建的平行目录树**。

隔离不靠「过滤算子清单」，而靠 **`if / else` 的结构互斥**：两棵树在同一份 CMake 里走互斥分支，物理上不可能同时进入构建图。

## 2. 与既有 overlay 机制的关系

| | overlay（单算子叠加层） | experimental（本方案） |
| --- | --- | --- |
| 生效时机 | **运行期**：往已装 OPP 的 `vendors/` 里塞一份新库，靠 `load_priority` 抢先 | **编译期**：决定这次构建装哪些算子 |
| 改动对象 | 已装 CANN / wheel 内嵌 OPP | 构建产物 |
| 解决什么 | 只想替换**某个算子**的实现，不动其余 | 新算子还在验证期，不想进稳定包 |

两者正交，可以并存；本方案不依赖 overlay，也不改变 overlay 的任何行为。

## 3. 术语

| 词 | 含义 |
| --- | --- |
| **稳定树** | `fla/ops/ascendc/<域>/<算子>/`，`ENABLE_EXPERIMENTAL=OFF` 时参与构建 |
| **实验树** | `experimental/fla/ops/ascendc/<域>/<算子>/`，`ENABLE_EXPERIMENTAL=ON` 时参与构建 |
| **域** | 算子的一级分类：本仓为 `cp` / `gdn` / `kda`（另有 `common`，只放公共头，不是算子） |
| **前缀规则** | `<op>_depends` 里的路径在实验模式下由框架统一加 `experimental/` 前缀 |

## 4. 目录约定（本方案的核心决策）

### 4.1 实验树整体镜像稳定树的仓库相对路径

```
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/          ← 稳定树
experimental/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/   ← 实验树（同构镜像）
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── chunk_fwd_o_def.cpp
└── op_kernel/
```

**为什么必须镜像，而不是 `experimental/<域>/<算子>/`**：

`func.cmake` 的前缀规则只做一件事——把 `<op>_depends` 的值前面拼 `experimental/`。本仓的 `depends` 写的是**仓根相对路径**（例如 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h`），所以：

```
experimental/ + fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h
  = experimental/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h
```

只有让实验树保持同一段相对路径，这个拼接结果才存在。若改成 `experimental/<域>/<算子>/`，就必须把全仓 `depends` 改写成域相对形式——那是更大的改动面。

> **代价**：目录层级多一层 `fla/ops/ascendc/`，可读性差一些。换来的是**算子源码与 CMakeLists 零改动**（`depends` 不用动、`add_subdirectory` 不用动）。

### 4.2 硬性约束

| 约束 | 原因 |
| --- | --- |
| **实验树内不要放 `experimental/CMakeLists.txt`（根层）** | 发现逻辑用 `GLOB_RECURSE .../experimental/CMakeLists.txt`，递归匹配任意深度的同名文件；根层那一份会被当成一个名叫 `experimental` 的「算子」 |
| **`experimental/<...>/common` 必须是稳定树的超集** | 框架只在实验模式下复制实验树那份 common 到 `src/common`，稳定树独有的头不会进来 → **静默缺头**（详见 §6.3） |
| **同一个算子的实验版与稳定版必须同名同结构** | 发现逻辑用目录名（`NAME`）推导算子名，两棵树的同名算子在各自模式内独立，不会冲突；但混放会 |
| 实验算子不能依赖「稳定树独有」的算子 | 依赖前缀会把它指到实验树，实验树里没有 → 被静默跳过（§6.2） |

## 5. 隔离是怎么保证的（四道闸）

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
   闸4 过滤  --ops=<op> 在已选集合里再做减法（不能把另一棵树拉进来）
```

**闸 1（`cmake/func.cmake` 的 `if/else`）是唯一的根本保证**：同一个 `if` 的两个分支，不是两条独立语句，所以两棵树不可能同时被 glob 到。

闸 2 本身不判断模式——它只是遍历绝对路径，树的信息已经编码在 `OP_DIR` 里（`…/experimental/...` 或 `…/fla/ops/ascendc/...`）。

### 5.1 四处已知边界（不是漏，是共享件）

| 边界 | 说明 | 影响 |
| --- | --- | --- |
| 根级 `common/` 恒编 | `custom_build.cmake` 里 `add_subdirectory(common)` 无条件执行；`experimental/` 下**没有** common 目录 | 实验包与稳定包共用同一份根级 common，改动会同时影响两者 |
| 依赖可回退到顶层 | 写 `../xxx` 会归一化回顶层（见 §6.2） | 是**合法手法**，但要显式知道自己在借顶层的东西 |
| `examples/` 按 `ASCEND_OP_NAME` 恒加 | 与模式无关 | `--ops` 命中 examples 目录就会加进来 |
| `--run_example` 的目录排除 | 非实验模式排除 `*/experimental/*`（`build.sh:621-623`） | 防止拿稳定包去跑实验 example |

## 6. 前缀规则（`func.cmake`）

### 6.1 规则本体

```cmake
if (ENABLE_EXPERIMENTAL)
    set(depend_info_update "experimental/${depend_info}")
else()
    set(depend_info_update ${depend_info})
endif()
```

出现两处，处理对象不同、互不重叠：

| 位置 | 处理对象 | 产物 |
| --- | --- | --- |
| `cmake/func.cmake:160` | 被依赖**算子**的目录 | `OP_DEPEND_DIR_LIST` → 补挂构建图 + 隐式扩编译范围 |
| `cmake/func.cmake:540` | 被依赖**算子**的 kernel 源 copy（**排除 common**） | `add_ops_src_copy` target |

> `func.cmake:540` 附近有 `if("<dep 名>" STREQUAL "common") continue()`，所以 common 不走这条；common 由 `custom_build.cmake:1152-1159` 单独处理（切换 `COMMON_SRC_DIR`）。

### 6.2 两条容易踩的路径语义

**① `../` 会归一化回顶层**

```
experimental/ + ../fla/ops/ascendc/gdn/...  →  fla/ops/ascendc/gdn/...
```

这是 CMake 的路径语义，不是特例。想「在实验模式下故意复用稳定树的某个依赖」，就写 `../` 前缀。

**② 解析不到是静默跳过**

```cmake
if (NOT EXISTS <结果>/op_host/CMakeLists.txt AND NOT EXISTS src/<结果>/CMakeLists.txt)
    continue()          # 不打印任何 message
endif()
```

**构建照样成功**，只是少挂一个算子。所以实验树的依赖必须自带齐。

### 6.3 common 的静默缩水风险

框架在实验模式下只把 `experimental/<...>/common` 复制进 `src/common`。如果实验树的 common 是**分叉副本**（而不是稳定树的超集），稳定树独有的头文件就不会进 `src/common`，坑在于**目录存在、复制成功、只在编译期报缺头**。

> 参考仓 `cann/ops-transformer` 就有这个问题：`attention/common` 稳定 126 / 实验 97（共享仅 47）、`gmm/common` 稳定 79 / 实验 1（共享 0），三者**都不是镜像**。
>
> 本仓当前 `experimental/<...>/common` 尚未出现；一旦引入，必须保证是稳定树的超集。

## 7. 本次修改清单

| # | 文件 | 改动 | 理由 |
| --- | --- | --- | --- |
| 1 | `cmake/func.cmake:70-84` | 实验分支从「按域列白名单」改为与 `else` 分支对称的一条 `GLOB_RECURSE …/experimental/CMakeLists.txt` | 原白名单是参考仓的域（`ffn/gmm/moe/posembedding/...`），本仓一个都不存在；且其中一行把**算子名当域**用。改成递归匹配后，**新增实验算子 / 新增实验域都不需要再改 cmake** |
| 2 | `cmake/custom_build.cmake:292-299` | 删除 `add_subdirectory(experimental/attention)` 整块 | 本仓没有 attention/mhc/mamba 这类模块聚合器分类，该目录不存在，`add_subdirectory()` 指向不存在目录是 **CMake fatal error** |
| 3 | `cmake/scripts/prepare.sh` | 为 `ENABLE_EXPERIMENTAL` 补默认值 `OFF` | 它是全链**唯一没有默认值**的变量；二次 configure 若取到空串，两次配置的算子集就可能不一致 |
| 4 | `scripts/ci/parse_changed_ops.py` | 路径解析改为**树根感知 + 算子目录标记定位**：识别 `fla/ops/ascendc/` 与 `experimental/fla/ops/ascendc/` 两种根；用 `op_host / op_kernel / tests / …` 标记定位算子目录边界；域清单改为 `cp / gdn / kda`，非算子目录（`common / 3rd / tools / tests / internal_ops`）直接忽略 | 原实现假设路径是 `<域>/<算子>/…`（2 层）或 `experimental/<域>/<算子>/…`（3 层）；本仓是 5/6 层且**算子层级不固定**（`kda/chunk_kda_fwd` 2 层、`gdn/chunk_gdn_fwd/chunk_fwd_o` 3 层、`kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post` 4 层），按固定下标取会把 `fla` 当域、`ops` 当算子 |
| 5 | `experimental/README.md` | 新增：目录约定 + 使用方式 | 让「实验目录怎么放、怎么编」在仓库里可见 |

### 7.1 未修改的（有意）

| 项 | 原因 |
| --- | --- |
| 3 个 `op_host/CMakeLists.txt` 里已有的 `experimental/${dep}` 手写前缀 | 在 §4.1 的镜像约定下，它与框架的前缀**结果一致**，不冲突；保留可读性（显式优于隐式）。若将来改成域相对 depends，需一并删除 |
| `cmake/scripts/util/ascendc_impl_build.py` | 该脚本用**相对探测**解析 kernel 源（`get_kernel_source` / `_get_kernel_source`），不硬编码树根；只要实验树内部结构同构即无需改动 |
| `build.sh` | `--experimental` 解析、`-DENABLE_EXPERIMENTAL=TRUE` 透传、`--pkg-type` 互斥检查均已存在，无需改 |

## 8. 使用与验证

```bash
# 稳定包（默认）
bash build.sh --pkg --soc=ascend910b --ops=chunk_fwd_o

# 实验包：从 experimental/ 树里选算子
bash build.sh --pkg --experimental --soc=ascend910b --ops=<实验算子名>

# 实验包：全量实验算子
bash build.sh --pkg --experimental --soc=ascend910b
```

**验证要点**（本 PR 未在真机跑，需 CI/本地补）：

1. 只配不编：`cmake -DENABLE_EXPERIMENTAL=TRUE …` 必须在**没有** `experimental/` 目录时也**不报错**（递归 glob 匹配空集是合法行为）。
2. 反向：`-DENABLE_EXPERIMENTAL=FALSE` 时 `experimental/` 完全不参与（可在 `experimental/` 下放一个故意写错的 `CMakeLists.txt`，若构建仍成功即证明未被 include）。
3. 两棵树同名算子：验证不会 target 重名。
4. 依赖：实验算子声明一个实验树内依赖，确认被正确补挂；再声明一个只在稳定树有的依赖，确认被静默跳过（预期行为，但要知道）。

## 9. 未完成 / 后续项

| 项 | 说明 |
| --- | --- |
| **CI 实验 job** | 当前 CI（`.github/workflows/ci.yml`）没有任何 job 传 `--experimental`，等于这条链**从未被验证**。需新增一个独立 job（独立产物，不与稳定包合并） |
| **`parse_changed_ops.py` 未在 CI 验证** | 该脚本仅在 `--PR_PKG` + experimental 时被调用（`build.sh:1022`）；当前无实验 job，所以改动后也无实际执行路径。新增实验 job 时必须先验证它 |
| **common 镜像校验** | 建议加一个离线门禁：若存在 `experimental/<...>/common`，断言它是稳定树同名目录的超集 |
| **Python / wheel 侧** | `torch_custom/fla_npu/` 完全不感知 `--experimental`。实验算子要能被 Python 调到，需要：Stable-ABI 三件套（`csrc/src/stable_<op>.cpp` + `_stable.py` wrapper + `_ASCENDC_OPS` 一行）+ OPP 装载隔离决策（实验 OPP 不能顶掉稳定 OPP） |
| **静态库** | `--static` 与实验算子未做适配，暂不支持 |

## 10. 风险与坑

| 坑 | 后果 | 规避 |
| --- | --- | --- |
| 实验树根层放 `CMakeLists.txt` | 会多出一个名叫 `experimental` 的「算子」 | 约定禁止（§4.2） |
| 实验树 common 不是超集 | 编译期缺头，且不报「路径不存在」 | §6.3 |
| 依赖只在稳定树有 | 静默丢依赖，构建成功 | §6.2 |
| `--pkg-type=rpm/deb` 搭配 `--experimental` | 外层守卫已禁止；但注意 `--pkg-type=all` **会绕过**该检查（守卫条件是 `!= run && != all`） | 使用时自行避免 |
| 实验包与稳定包共用根级 `common/` | 改 common 会同时影响两者 | 改后两种包都要验 |
