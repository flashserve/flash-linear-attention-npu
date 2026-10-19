# experimental 适配：现状、实测结果与遗留问题

> **本文定位**：一份读完即可掌握「做到哪、验过什么、还差什么」的现状文档。
> 详细设计（代码骨架、隔离机制、前缀规则推导）见 [experimental 构建适配设计.md](experimental%20构建适配设计.md)。
> 目录约定与使用方式见 [`experimental/README.md`](../../experimental/README.md)。

| 项 | 值 |
| --- | --- |
| 分支 | `feat/experimental-build`（基于 `main`） |
| 改动规模 | 7 个文件，**+569 / −66** |
| 提交 | `98d1d84` → `7aa185a` → `9c76581` → `80a2011` → `239d26c` → `f5be7b4` |
| 实测环境 | A5 服务器，`ascend950` + CANN 9.2.0 + cmake 3.28.3，基线 `main@67dc99d7` + 本 PR 补丁 |
| 状态 | 机制已打通并实测出包；**遗留 1 个待决策项（G1）+ 若干后续项** |

---

## 1. 一页摘要

### 做了什么

把本仓从参考仓 vendored 过来的 `--experimental` 骨架**接到本仓的目录布局上**，让「实验树」真正可编。
参考仓的消费端写死了它自己的域（`ffn/gmm/moe/posembedding/…`）与目录结构，本仓一个都不匹配，所以该开关此前**完全不可用**（CMake fatal）。

### 核心设计决策（4 条）

| 决策 | 内容 | 理由 |
| --- | --- | --- |
| **目录镜像** | 实验树 = `experimental/` + **仓库相对路径**，即 `experimental/fla/ops/ascendc/<域>/<算子>/` | `<op>_depends` 写的是仓根相对路径，框架只负责前插 `experimental/`；只有镜像才能让拼接结果真实存在，从而做到**算子源码零改动** |
| **发现方式** | 递归 glob，不按域列白名单 | 新增实验算子 / 新增实验域都不必再改 CMake；且支持「稳定树还没有的新域」 |
| **依赖回退** | 实验树缺依赖时回退稳定树 + 打告警 | 否则依赖被静默丢弃，后果不是少编一个算子，而是**打包链接失败**（见 G2） |
| **开关恒下发** | `-DENABLE_EXPERIMENTAL` 每次都传 | 否则 CMakeCache 会残留上次的 `TRUE`，普通构建会继续编实验树 |

### 现在的能力边界

| 能 | 不能 |
| --- | --- |
| `--experimental` 发现实验树里的算子（含 3～4 层深嵌套） | 实验包与稳定包**同时**编（互斥，不是加法） |
| 实验算子单独编包并产出 `.run` | 实验树为空时配置（0 算子不支持，见 G8） |
| 依赖自动回退稳定树（带告警） | kernel 公共头隔离（**G1，待决**） |
| `--PR_PKG --experimental` 按变更文件定位实验算子 | Python/wheel 侧感知 `--experimental`（`torch_custom/` 完全不感知） |

---

## 2. 改动清单

| # | 文件 | 改动 | 类别 |
| --- | --- | --- | --- |
| 1 | `cmake/func.cmake` | 实验分支从「按域白名单」改为与 `else` 对称的递归 glob | 修复（原为 fatal 的上游成因） |
| 2 | `cmake/custom_build.cmake` | 删除 `add_subdirectory(experimental/attention)` | **修复 fatal**（目录不存在） |
| 3 | `cmake/func.cmake` | 新增 `fla_resolve_depend_path()`，替换两处内联前缀，实现**依赖回退** | 新增能力（G2） |
| 4 | `build.sh` | 删除「无匹配算子时兜底成 `fused_infer_attention_score`」 | **修复**（该算子本仓不存在） |
| 5 | `build.sh` | `-DENABLE_EXPERIMENTAL` 由「仅 TRUE 时追加」改为**恒下发** | **修复**（缓存残留） |
| 6 | `cmake/scripts/prepare.sh` | 为 `ENABLE_EXPERIMENTAL` 补默认值 `OFF` | 可读性（非 bug） |
| 7 | `scripts/ci/parse_changed_ops.py` | 路径解析改为「树根感知 + 算子目录标记定位」 | 修复（层级不固定） |
| 8 | `experimental/README.md` | 新增目录约定 | 文档 |
| 9 | `docs/architecture/experimental 构建适配设计.md` | 新增详细设计 | 文档 |

> 其中 3 处是**真正的缺陷修复**（#2 fatal、#4 指向不存在算子、#5 缓存残留），1 处新增能力（#3），其余为对齐与文档。

---

## 3. 实测结果

### 3.1 环境

| 项 | 值 |
| --- | --- |
| 机器 | A5 验证服务器 `179.228.3.2`（另一台 `179.223.2.2` 前期也验证过，结论一致） |
| 芯片 / 工具链 | `ascend950` / CANN 9.2.0（`/home/zzy1205/Ascend/cann-9.2.0`）/ cmake 3.28.3 |
| Python | conda `zzy`：3.12.15、torch 2.7.1+cpu、torch_npu 2.7.1.post6 |
| 机器规格 | 1.5 GB 内存 / 256 核 → **必须限并发**（见 §6） |

### 3.2 结果矩阵（14 项）

| # | 验证项 | 结果 | 关键证据 |
| --- | --- | --- | --- |
| 1 | `bash -n build.sh` / `prepare.sh`、`py_compile` | ✅ | 全部 OK |
| 2 | **未打补丁**代码 + `--experimental` | ✅ 复现原缺陷 | `add_subdirectory given source "experimental/attention" which is not an existing directory`（`custom_build.cmake:296`） |
| 3 | 打补丁 + 3 层深真实实验算子 | ✅ | configure 成功；`OP_DIR_LIST` **只含**该实验算子（隔离生效） |
| 4 | 隔离：实验树根放非法 `CMakeLists.txt` + **关闭**开关 | ✅ | configure 成功（证明未 include），cache=`FALSE` |
| 5 | 同上 + **打开**开关 | ✅ | `CMake Error at experimental/CMakeLists.txt:1: Parse error`（证明确实 include） |
| 6 | `--PR_PKG --experimental` → 3 层深算子 | ✅ | `rc=0`，输出 `Operators that need custom package compilation:kda_gate_bwd_post` |
| 7 | `--PR_PKG` 空清单 | ✅ | `rc=200`「No custom packages to build」 |
| 8 | `--PR_PKG` 只改稳定树路径（实验模式） | ✅ | `rc=200` 正确忽略 |
| 9 | 兜底算子 `fused_infer_attention_score` | ✅ 已消失 | 三个 `--PR_PKG` 用例中均出现 0 次 |
| 10 | 有 tiling 源、无依赖的实验算子全量编包 | ✅ | `rc=0`，产出 `build_out/fla_npu_linux-x86_64.run` |
| 11 | 有依赖但**未**带依赖 | ❌ → 已定位为 G2 | `cannot find -lophost_transformer_tiling_obj`；`chunk_kda_bwd_tiling.cpp` 编译 **0 次** |
| 12 | 有依赖且**带齐**依赖闭包 | ✅ | `rc=0`；`*_tiling.cpp` 编译 **20 次**；3 个 `aic-*-ops-info.ini`；产出 `.run`；从 `experimental/` 路径编译出 **43 个 `.o`** |
| 13 | **G2 修复后**：实验树**只放算子、不放依赖** | ✅ **关键** | `rc=0`；回退告警 **1 次**且精确指向 `fla/ops/ascendc/kda/chunk_kda_bwd`；`chunk_kda_bwd_tiling.cpp` 编译 **4 次**（修复前 0 次）；缺 `tiling_obj` 库 **0 处**（修复前 1 处、直接失败）；产出 `.run` |
| 14 | OFF 模式回归（同样场景） | ✅ | configure 成功，回退告警 **0 次**（OFF 逻辑与修改前逐字符一致） |

### 3.3 产物内容核实（第 12/13 项）

`build_out/fla_npu_linux-x86_64.run` 的暂存目录 `vendors/fla_npu_transformer/` 下确认含实验算子：

```
dynamic/kda_gate_bwd_post.py                     ← 生成的动态 Python
op_proto/inc/kda_gate_bwd_post_proto.h           ← proto 头
kernel/config/<soc>/kda_gate_bwd_post.json       ← kernel 配置
kernel/<soc>/kda_gate_bwd_post                   ← kernel
op_impl/ai_core/tbe/<vendor>_impl/ascendc/kda_gate_bwd_post/
op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so
op_api/lib/libcust_opapi.so
```

### 3.4 关键因果链（G2 的取证）

同一个算子 `kda_gate_bwd_post`（`depends` = `kda/chunk_kda_bwd`，后者含 `chunk_kda_bwd_tiling.cpp`）：

| 模式 | 依赖是否补挂 | `chunk_kda_bwd_tiling.cpp` 编译 | `ophost_transformer_tiling_obj` | 结果 |
| --- | --- | --- | --- | --- |
| 稳定树 | ✅（日志 274 处） | **4 次** | 被创建 | ✅ 打包成功 |
| 实验树（修复前） | ❌ 静默丢弃 | **0 次** | **从未创建** | ❌ `cust_opmaster` 链接失败 |
| 实验树（修复后） | ✅ 回退稳定树 | **4 次** | 被创建 | ✅ 打包成功 |

> 结论：`<op>_depends` 的静默丢弃**不是「少编一个算子」，是打不出包**。这就是把 G2 从「建议」提为「必修」的依据。

---

## 4. 遗留问题清单

| ID | 问题 | 状态 | 影响面 | 建议 |
| --- | --- | --- | --- | --- |
| **G1** | **kernel 公共头未隔离** | 🔴 **待决策** | 26 个算子 / 68 处引用 | 切换 + 超集门禁（§4.1） |
| G2 | 依赖无回退 | 🟢 **已修复**（`f5be7b4`） | — | 见 §3.4 |
| G3 | CI 没有实验 job | 🔴 未做 | 整条链只能手工验证 | 新增独立 job（独立产物） |
| G4 | `tests/test_config.yaml` 不存在 | 🟡 既有问题，未动 | `build.sh:1206` 的非实验 `--PR_PKG` 分支也不通 | 单独排期 |
| G5 | `ascendc_impl_build.py` 看似少适配 | 🟢 已澄清**不需要改** | — | 它用相对探测而非模式分支 |
| G6 | `--run_example` 不排除 `experimental/` | 🟡 未做 | 目前实验算子少，影响小 | 1 行改动即可对齐参考仓 |
| G7 | `--soc=ascend950` 时 kernel 落到 `build/binary/ascend910b/` | 🟡 既有问题，未动 | 影响「950 包是否含 950 kernel」 | 查 `BINARY_COMPUTE_UNIT` 取值来源 |
| G8 | 实验树为空不可配置 | 🟢 已确认并写入文档 | 空树 → 0 算子 → 无 `ops-info.ini` → `OpFileNotExistsError` | 可选：换成更友好的报错 |
| G9 | 往实验树 `cp -r` 会嵌套复制 | 🟢 已在文档标注 | 同名算子重复 → autogen 冲突 → `ops-info.ini` 生成失败 | 文档给正确做法 |
| G10 | `--static` 未与实验算子适配 | 🟡 未做 | — | 后续 |
| G11 | Python / wheel 侧不感知 `--experimental` | 🟡 未做 | 实验算子无法从 Python 调用 | Stable-ABI 三件套 + OPP 装载隔离决策 |

### 4.1 G1 详细分析（唯一待决策项）

#### 现状事实

```
fla/ops/ascendc/common/                        ← 只有这一份，7 个 .hpp
├── kernel_utils/block/block_mmad_pingpong_tla*.hpp     (5 个变体)
├── kernel_utils/tile/copy_l0c_to_ub.hpp
└── kernel_utils/vector/regbase.hpp

experimental/fla/ops/ascendc/common/           ← 不存在
```

| 事实 | 数值 |
| --- | --- |
| 引用该头的算子 | **26 / 36（72%）** |
| 引用处数 | **68 处** `#include "kernel_utils/..."`
| 是否出现在任何 `<op>_depends` 里 | **0 次** —— 它不是依赖图的一部分，而是**构建级 include 根** |
| 控制点 | `cmake/config.cmake:41` 一个变量 `OPS_ADV_UTILS_KERNEL_INC` |
| 消费者 | kernel 编译 include 根、复制到 staging（`func.cmake:403/411`）、打进包（`custom_build.cmake:952-962`、`CMakeLists.txt:548-551`） |

#### 问题

`--experimental` 对这份头**完全无效**。后果：

- 改 `fla/ops/ascendc/common/` 下任何一行 → **稳定包与实验包同时被改**
- 实验算子若需要「实验版的公共头」，没有任何手段

#### 为什么它应该隔离

算子的隔离靠 `func.cmake` 的 `if/else` 决定「编哪棵树」；公共头是**所有算子共享的构建级输入**，理应由同一个开关决定「用哪份头」。一次构建只编一棵树，所以「头跟着切换」自洽，同一构建内不存在混用。

#### 两种修法的取舍

| 方案 | 做法 | 是否真隔离 | 风险 |
| --- | --- | --- | --- |
| A. **切换**（推荐） | `config.cmake:41` 在实验模式下指向 `experimental/fla/ops/ascendc/common` | ✅ 是 | 若实验 common 是**不完整分叉副本** → 编译期缺头。参考仓正是这样坏的（`attention` 共享 47/126、`mc2` 共享 1/242、`gmm` 共享 0/79） |
| B. 叠加 | 稳定头先拷、实验头后覆盖 | ❌ 否（实验算子仍看得见稳定头） | 无缺头风险，但不是隔离 |

**推荐 A + 超高门禁**：把参考仓的「静默缺头」变成「显式失败」。

```cmake
# cmake/config.cmake:41 附近
if (ENABLE_EXPERIMENTAL AND EXISTS "${OPS_ADV_DIR}/experimental/fla/ops/ascendc/common")
    # 门禁：实验 common 必须是稳定 common 的超集，否则 FATAL_ERROR 并列出缺失文件
    file(GLOB_RECURSE _stable_hdrs RELATIVE "${OPS_ADV_DIR}/fla/ops/ascendc/common"
         "${OPS_ADV_DIR}/fla/ops/ascendc/common/*")
    set(_missing)
    foreach(_h ${_stable_hdrs})
        if (NOT EXISTS "${OPS_ADV_DIR}/experimental/fla/ops/ascendc/common/${_h}")
            list(APPEND _missing "${_h}")
        endif()
    endforeach()
    if (_missing)
        message(FATAL_ERROR "experimental common 不是稳定树的超集，缺少: ${_missing}")
    endif()
    get_filename_component(OPS_ADV_UTILS_KERNEL_INC
        "${OPS_ADV_DIR}/experimental/fla/ops/ascendc/common/" REALPATH)
else()
    get_filename_component(OPS_ADV_UTILS_KERNEL_INC
        "${OPS_ADV_DIR}/fla/ops/ascendc/common/" REALPATH)
endif()
```

改动量约 12 行，**全在 `config.cmake` 一处**（三处消费者都用同一个变量，改一处即一致）；无实验 common 目录时走原路径，**默认行为不变**。

#### 隔离后的必然后果（需接受）

这份头会被打进包。隔离后：实验包与稳定包的 kernel 头不同 → **不能用稳定包复现实验包的问题**。这正是隔离的代价，也是它想要的效果。

---

## 5. 未验证项

| 项 | 原因 |
| --- | --- |
| CI 实验 job 下的完整链路 | CI 里没有该 job（G3） |
| 两棵树放**同名**算子 | 未构造该用例；从 target 命名机制看不会冲突，但没实测 |
| G1 的三种场景（无实验头回归 / 缺文件门禁 / 超集隔离） | G1 尚未实现 |
| Python 侧真正调用实验算子 | G11 未做 |

---

## 6. 复现方法

### 6.1 打补丁到最新 main

```bash
# 本地
git -C fla-built-in-gpt/experimental-build diff <base>..HEAD --output=/tmp/pr.patch
# 服务器
cd ~/flash-linear-attention-npu
git checkout -b verify/exp main
git apply /tmp/pr.patch
```

> 注意：`git diff` 输出**不要经 PowerShell `>` 重定向**（会写成 UTF-16，`git apply` 报 `No valid patches`）。用 git 自己的 `--output`。
> 内联 `ssh "...命令..."` 会把引号吞掉。**统一用「本地写脚本 + scp + bash 执行」**。

### 6.2 构造实验树（关键：别用 `cp -r` 拷目录）

```bash
# 只拷最上层算子；internal_ops/* 会随之带进来
SRC=fla/ops/ascendc/kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post
DST=experimental/fla/ops/ascendc/kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post
mkdir -p "$DST" && cp -a "$SRC/." "$DST/"     # 拷内容，不拷目录
```

> `cp -r <父> <已存在的目标目录>` 会把父目录**嵌套进自己内部**（出现 `chunk_kda_bwd/chunk_kda_bwd/...`），产生同名算子重复 → autogen 目标互相覆盖 → `ops-info.ini` 生成失败 → configure 报 `ops prepare build failed`。

### 6.3 构建（这台机器必须限并发）

```bash
source ~/Ascend/cann-9.2.0/ascend-toolkit/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh && conda activate zzy

bash build.sh --pkg --soc=ascend950 --experimental --ops=kda_gate_bwd_post -j4
```

> 机器为 **1.5 GB 内存 / 256 核**。默认并行度会 OOM（实测出现 `Error 137`，并曾把机器搞重启一次）。
> 注意 `cmake/scripts/prepare.sh:13` 的并行度写死为 `nproc*2`（=512）且**没有覆盖开关**；`-j` 只作用于主构建阶段。

### 6.4 关键中间量自查

想在 configure 阶段确认「实验算子到底有没有被发现 / 依赖有没有回退」，可临时插打印：

```cmake
# cmake/custom_build.cmake，op_add_subdirectory(OP_LIST OP_DIR_LIST) 之后
message(STATUS "VERIFY_OPDIR=${OP_DIR_LIST}")
# 依赖遍历 endforeach 之后
message(STATUS "VERIFY_DEPDIR=${OP_DEPEND_DIR_LIST}")
```

---

## 7. 相关文档

| 文档 | 内容 |
| --- | --- |
| [experimental 构建适配设计.md](experimental%20构建适配设计.md) | 详细设计：代码骨架与调用关系、四道闸、`OP_DIR_LIST` 来源与消费、前缀规则、参考仓机制差异 |
| [`experimental/README.md`](../../experimental/README.md) | 目录约定、三条硬性约束、使用方式、已知限制 |
| [适配层接入指南.md](适配层接入指南.md) | Python 适配层（G11 涉及） |
