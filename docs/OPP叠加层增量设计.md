# FLA NPU 算子包「单算子叠加层」增量设计

- **作者**：zzy1205
- **评审人**：待定（改动落在 `torch_custom` 的加载/安装脚本；不涉及算子 `def`/`aclnn`/`torch` 接口入参，按仓库 CODEOWNERS 不属 ABI 敏感改动）
- **状态**：已实现并通过 **A5 + A2 双平台**行为验证（A5：Ascend950PR / CANN 9.2.0-beta.2；A2：**910B3 真机 / aarch64** / CANN 9.2.0）；开发分支 `feat/opp-delta-overlay`
- **日期**：2026-10-14
- **验证平台**：A5 = CANN `9.2.0-beta.2` / Ascend950PR；A2 = CANN `9.2.0` / Ascend910B3（aarch64，8 卡）；A3（`ascend910_93`）仅交叉编译

---

## 1. 背景与问题

### 1.1 现状

`fla_npu` 把**整份 OPP** 嵌在 wheel 内：`site-packages/fla_npu/opp/vendors/fla_npu_transformer/**`（35 算子 / 1265 文件），host 库按**包内绝对路径**加载（`__init__.py` 的 `_resolve_vendor_dir()` + `relative_to(package_dir)`，明确拒绝外部 CANN vendor 目录）。

因此单算子迭代的代价极高：

| 现状 | 事实 |
|---|---|
| 产物粒度是整份 OPP | `libcust_opapi.so`、`aic-<soc>-ops-info.json`、`binary_info_config.json` 都**按算子集聚合**（实测：1 算子 71,824 B / 2 个 aclnn 符号；35 算子 1,794,712 B / 74 个） |
| 官方"局部覆盖安装"殃及其余算子 | CI 分项 06（`ci/run_checks.sh` 的 `check_scoped_wheel_opp_install()`）用单算子 run 包 `--full` 覆盖安装，安装器随后打印 `Operator support status after installing this run package:` 并 **WARNING 其余算子不在本次范围内** |
| 无回滚、无并存 | 覆盖是 `rmtree + copytree` 整树替换 |
| 迭代成本 | 改一个算子要重出整份 OPP：**实测 884 s（14m45s）/ 101 MB** |

### 1.2 根因：运行期"算子路径"的查找模型

运行期要回答"某个算子的东西在哪"，实际是**四条互相独立的链**：

| 链 | 谁在查 | 要回答的问题 | 查什么（按顺序） |
|---|---|---|---|
| **L1 host 符号** | Python 进程 | `aclnnXxx` 这个函数在哪？ | 库列表**逐个 `dlsym`**：`<overlay>/op_api/lib/libcust_opapi.so` → `<base>/…` → CANN `libopapi.so`，先命中先用。**例外**：Stable-ABI launcher 只用 `FLA_NPU_OP_API_LIB` 指向的那**一个**库 |
| **L2 算子注册** | CANN 运行时 | 这个算子存不存在、什么规格？ | 逐个 OPP 树条目查 `op_impl/ai_core/tbe/config/<soc>/aic-<soc>-ops-info.json` |
| **L3 kernel 选择** | CANN 运行时 | 该算子该跑哪个二进制？ | 同树 `op_impl/ai_core/tbe/kernel/config/<soc>/binary_info_config.json`（顶层键就是算子名） |
| **L4 实体** | CANN 运行时 | 二进制/切分/原型在哪？ | `.../kernel/<soc>/<op>/*.o`、`.../op_tiling/liboptiling.so`、`op_proto/**` |

树的顺序由 `ASCEND_CUSTOM_OPP_PATH` 决定（`overlay → base → built-in`），FLA 在 `import` / `.pth` 阶段把它 prepend 进去。

**关键性质：查找是"逐算子 / 逐符号"，而索引与库是"按算子集聚合"。**

| 产物 | 单算子包（`chunk_fwd_o`） | 整包（35 算子） |
|---|---|---|
| ① `config/<soc>/aic-<soc>-ops-info.json` | 3,762 B / **1 个算子** | 257,299 B / **35 个** |
| ② `kernel/config/<soc>/binary_info_config.json` | 3,822 B / **1 条** | 196,856 B / **35 条** |
| ③ `op_api/lib/libcust_opapi.so` | 92,304 B / **2 个 aclnn 符号** | 1,794,712 B / **74 个** |
| ④ `kernel/<soc>/<op>/*.o` | **4** 个 | **187** 个 |

- **逐项查找** ⇒ 多份来源可并存、可回退 —— **这是叠加层能成立的前提**；
- **聚合产物** ⇒ 替换 ①②③ 中任何一个 = **一次性改掉整批算子的"路径答案"**。

**这就是"局部覆盖安装殃及其他算子"的机制**：覆盖动作是 `rmtree` 包内 `fla_npu_transformer` + `copytree` 单算子树，于是对**其余 34 个算子**——② 清单里没有它（未注册）、③ 索引里没有它（选不到 kernel）、① 库里没有它的符号（host 解析失败，launcher 路径直接 `symbol not found`）、④ 也没有它的 `.o`。

## 2. 目标

1. **单算子增量**：改一个算子只需**只编该算子**，产出可安装的小包；安装后该算子生效、**其余算子照常可用**。
2. **一键回滚**：一条命令回到叠加前状态。
3. **零构建系统改动**：沿用 `--ops=<op>` 只编一个算子，不引入 `--incremental`、不复活"跳过构建"开关。
4. **默认行为不变**：未安装叠加层时，加载与安装行为与改动前一致。
5. **可观测、可审计**：加载顺序、安装状态、生效层（host / kernel）都能被命令行直接看到。

## 3. 非目标

- ❌ 不做**对象级增量编译**（复用 `build/**/*.o`）。**说明**：本设计的"增量"在**编译范围**（`--ops`）与**安装方式**（叠加层），不是对象复用。仓库当前以测试固化"每次全量、自动清理"（`build.sh` 的 `set_env; clean; clean_build_out`；`tests/test_wheel_environment.py` 断言不得出现 `FLA_NPU_INCREMENTAL_BUILD`/`FLA_NPU_SKIP_RUN_BUILD`/`--incremental`）。**该契约是可改的策略、不是技术限制**，路线与实测见 §7.4。
- ❌ 不做**多版本并存/灰度**（仅需"叠加 + 回滚"）。
- ❌ 不改 **run 包自身安装器**（`scripts/package/**`）：CI 分项 06 的断言对象必须保持原样。
- ❌ 不重写 pip `RECORD`：叠加层不是 pip 管理的文件集，回滚走 `--remove-overlay`。
- ❌ 不覆盖 **launcher 独占算子**（`_LAUNCHER_ONLY_OPS`）。
- ❌ 不做 A2/A3 的**运行**验证（开发机为 A5，无对应硬件）。

## 4. 需求

### 4.1 功能性

| 编号 | 需求 | 落地 |
|---|---|---|
| F1 | 单算子包可作为**叠加层**装入包内，不触碰 base vendor | `install_opp.py --overlay --source <vendor_dir>` |
| F2 | 加载时叠加层**优先于** base，base 优先于 CANN | 库列表 `[overlay, base, CANN]` + `ASCEND_CUSTOM_OPP_PATH` 顺序 |
| F3 | 叠加层未提供的算子**自动回落** base | `_runtime.symbol()` 按符号逐个库查找（既有语义，无需改动） |
| F4 | 一键回滚 | `install_opp.py --remove-overlay` |
| F5 | 状态可查 | `install_opp.py --status`（各 vendor、文件数、算子名、`load_priority`） |
| F6 | 解释器启动阶段即就绪（规避 issue #429） | `fla_npu_opp_env.py`（`.pth`）按 `overlay → base` 预置环境 |
| F7 | 支持多个叠加层（有序） | 加载端**自动发现**包内 `fla_npu_overlay_*` 目录，按 `config.ini` 的 `load_priority` 排优先级；`FLA_NPU_OVERLAY_VENDOR`（`os.pathsep` 分隔）可显式覆盖，靠前者优先 |

### 4.2 非功能性（均为本机实测）

| 编号 | 指标 | 目标 | 实测 |
|---|---|---|---|
| N1 | 单算子增量编译耗时 | ≪ 整包 | **30–63 s**（对照整包 **884 s**，约 **14–29×**） |
| N2 | 叠加包体积 | ≪ 整包 | **1.3 MB**（对照 101 MB） |
| N3 | 叠加层文件数 | 只含该算子 | **339–346**（对照 base 1265） |
| N4 | 安装/卸载耗时 | 秒级 | 秒级（340 文件拷贝 / `rmtree`） |
| N5 | 其余算子行为 | **零变化** | 10 个算子级用例 rc 逐项一致；`FastGeluCustom`/`ChunkFwdH` 输出 sha256 完全一致 |
| N6 | 生效可验证（kernel / host 双侧） | 可精确预测 | kernel×3 + host×2 → 输出 `sum=2304 → 13824`（**正好 6 倍**） |
| N7 | 回滚后可复现基线 | 逐字节一致 | 回滚后单算子测试日志与基线**逐字节一致**（仅时间戳/PID 不同） |
| N8 | 生效时延 | 需重启进程 | 同进程不生效；**新进程**生效（库/符号已缓存） |
| N9 | **安装动作不得触碰 base** | base 逐字节不变 | 装/卸前后 base 目录指纹、`libcust_opapi.so`、ops-info md5 **全程不变**；除叠加层目录外仅 `config.ini` 一行被改写 |
| N10 | 对象复用（方案 D）潜在收益 | 决策参考 | 热树 + 改 1 个 host `.cpp` → **3 s**（省 90%）；改 **kernel** 源文件 → **0 次 opc**（kernel 目标未接依赖，当前不可用，见 R9） |

## 5. 方案总览

![叠加层架构与数据流](images/opp-overlay-arch.png)

```text
site-packages/fla_npu/opp/vendors/
├── config.ini                        load_priority=fla_npu_overlay_transformer,fla_npu_transformer
├── fla_npu_overlay_transformer/   ← 叠加层：只含本次改动的算子（339–346 文件）
│   ├── op_api/lib/libcust_opapi.so                       host 侧
│   ├── op_impl/…/config/<soc>/aic-<soc>-ops-info.json    清单（只列该算子）
│   ├── op_impl/…/kernel/<soc>/<op>/*.o                  device kernel
│   └── op_proto/、op_impl/…/op_tiling/                  proto / tiling
└── fla_npu_transformer/           ← 老整包（35 算子，一个字节不动）
```

**原理**：`_runtime.symbol()` 本来就是"**按符号逐个库找、先命中先用、miss 落下一个库**"。本设计把库列表从"1 个 custom"扩成"**有序多个** custom"，因此**运行时查找逻辑一行都不用改**；kernel 侧由 CANN 按 `ASCEND_CUSTOM_OPP_PATH` 顺序索引（叠加层在前）。于是 §1.2 的四条链全部变成"**有序多来源 + 逐项回落**"，其余算子不再受牵连。

## 6. 详细设计

### 6.1 模块划分与改动清单

| 模块 | 文件 | 职责 | 本次改动 |
|---|---|---|---|
| 路径解析 | `fla_npu/__init__.py` | 解析包内 vendor（仍强制包内），给出**有序**列表 | 新增 `overlay_vendor_names()`、`_resolve_vendor_dirs()`；保留 `_resolve_vendor_dir()` |
| 环境预置 | 同上 `_prepare_embedded_opp()` | 反序 prepend `ASCEND_CUSTOM_OPP_PATH`；设置库相关变量 | 由"单目录"改为"按 overlay→base 的列表" |
| 库装载 | 同上 `load_ascendc_opapi_libraries()` | 返回 `[overlay, base, CANN]` | 由"1 个 custom"改为"N 个 custom" |
| 符号解析 | `fla_npu/ops/ascendc/_runtime.py` `symbol()` | 按符号遍历库列表 | **未改动**（设计前提） |
| 稳定启动器 | `libfla_npu_stable.so` + `ops/ascendc/_stable.py` | 从 `FLA_NPU_OP_API_LIB` 指向的**单个**库解析全部符号 | 未改库；改为"指向完整库 + 有叠加层时切 ctypes" |
| 安装器 | `fla_npu/install_opp.py` | 装/卸/查叠加层；维护 `config.ini` | 新增 `--overlay` / `--remove-overlay` / `--status` |
| 启动钩子 | `fla_npu_opp_env.py`（`.pth`） | 启动阶段预置环境 | 支持 overlay→base 顺序 |

### 6.2 关键数据结构与四条不变量

| 名称 | 类型 | 语义 |
|---|---|---|
| `vendor_dirs` | 有序 `list[Path]` | `[overlay…, base]`；靠前者优先；全部经 `relative_to(package_dir)` 校验 |
| `_ASCENDC_OPAPI_LIBRARIES` | 有序 `list[CDLL]` | `[overlay_lib, base_lib, libopapi.so]`；`symbol()` 的查找域 |
| `FLA_NPU_OP_API_LIB` | str | **被 Stable-ABI launcher 消费**：必须指向**完整**算子集（base） |
| `FLA_NPU_OP_API_LIBS` | str（pathsep 列表） | 新增：完整有序库列表，供诊断 |
| `FLA_NPU_STABLE_ABI` | str | `ctypes` 强制参考路径；**装叠加层且用户未显式指定时自动置位** |
| `FLA_NPU_OVERLAY_VENDOR` | str（pathsep 列表） | 叠加层目录名；**可选**——未设置时从 `config.ini` 的 `load_priority` 与包内 `fla_npu_overlay_*` 目录自动发现 |
| 叠加层命名约定 | 前缀 `fla_npu_overlay_` | 带此前缀的包内 vendor 目录被视作叠加层：在 `config.ini` 中**一律排在 base 之前**，且被加载端自动发现（避免"装了但不生效"） |
| `opp/vendors/config.ini` | 文本 | `load_priority=<overlay>,<base>`（按优先级写，不再按名字排序） |
| `aic-<soc>-ops-info.json` / `binary_info_config.json` | JSON | **按算子集聚合**；叠加层只列自己带的算子（CANN 侧按多份索引并集查找，已实测） |

**四条不变量（评审重点）**

1. **`FLA_NPU_OP_API_LIB` 恒指"完整库"（base）**：launcher 从这一个库解析全部 aclnn 符号；指成单算子叠加层会让其余算子运行期报 `stable launcher: symbol not found: aclnnXxxGetWorkspaceSize`（开发中实际踩到并修复，见 §9.1）。
2. **装叠加层时后端自动切 ctypes**（可被 `FLA_NPU_STABLE_ABI` 覆盖）：launcher 是"一个算子绑一个库"，无法被叠加层覆盖；ctypes 路径才按符号遍历 `[overlay, base, CANN]`。
3. **叠加层只在包内**，不放宽 `relative_to(package_dir)` 校验（未装叠加层时行为与改动前一致）。
4. **不触碰 run 包安装器、不重写 pip `RECORD`**；回滚统一走 `--remove-overlay`。

### 6.3 加载来源的保护：`relative_to(package_dir)` 校验

`fla_npu` 在**两处**成对校验"路径必须落在包内"：

| 位置 | 代码 | 校验对象 |
|---|---|---|
| `__init__.py` `_resolve_vendor_dir()` | `resolved = vendor_dir.resolve()` → `resolved.relative_to(package_dir)`；抛 `ValueError` 就走不到 `return`，最终 `FileNotFoundError` | 包内 vendor 目录 |
| `__init__.py` `_resolve_packaged_opapi()` | 对 `…/op_api/lib/libcust_opapi.so` 再做一次 | 被 `ctypes.CDLL` 加载的 host 库 |

机制：`.resolve()` **会解析软链**，所以"包内软链指向包外"同样被挡；失败信息原文写着 *"External CANN vendor directories are not used as a fallback."*

**在保护什么**：

1. **加载来源的确定性**（最实在）：保证运行时加载的 OPP 就是这个 wheel 自带的那一份，否则语义随环境漂移、无法判断"跑的是哪份算子"。
2. **阻断环境噪声注入**：`libcust_opapi.so` 是直接 `CDLL` 进 Python 进程的本地代码；允许任意外部路径等于让任何能写那个目录的人影响该进程。属**卫生 / 供应链边界**，非沙箱级硬边界。
3. **与"CANN 侧发现"解耦的双轨设计**：包内 OPP 通过 `ASCEND_CUSTOM_OPP_PATH` 暴露给 CANN（host/tiling/kernel 发现），但 **host 库加载路径刻意不从环境取**（源码注释要求不要把 `op_api/lib` 加进 `LD_LIBRARY_PATH`）。
4. **由此得到的硬约束**：外部 vendor 的 **host 改动永远无法**靠环境变量生效（transformer 侧实测：外部 vendor 只对 **kernel** 层生效）→ 叠加层**必须放进包内**。本设计**保留**该校验。

### 6.4 核心流程

**流程 1：import 时建立加载域**

```text
import fla_npu
  ├─ _run_import_guards()
  ├─ _warn_if_embedded_opp_not_preconfigured()      # issue #429 提示
  └─ load_ascendc_opapi_libraries()
        ├─ _prepare_embedded_opp()
        │     ├─ _resolve_vendor_dirs()              → [overlay, base]（只含存在且包内的）
        │     ├─ 反序 prepend ASCEND_CUSTOM_OPP_PATH  → overlay, base, …
        │     ├─ FLA_NPU_OP_API_LIB = 完整库(base)
        │     └─ 有 overlay 且未显式指定 → FLA_NPU_STABLE_ABI=ctypes
        └─ 装载：先 libopapi.so（CANN 独立初始化），再 overlay_lib、base_lib
              → 返回 [overlay_lib, base_lib, libopapi.so]
```

**流程 2：一次算子调用的解析**：见 §1.2 四条链 —— L1 走 ctypes 的逐符号查找（`_runtime.symbol()`）或 launcher 的单库查找；L2–L4 由 CANN 按 `ASCEND_CUSTOM_OPP_PATH` 顺序逐项查。

**流程 3：安装 / 回滚**

```text
# 编（只编该算子；不引入 --incremental）
bash build.sh --pkg --soc=ascend950 --ops=<op> -j64        # 30–63 s
# 取 vendor 树（run 包不支持 makeself --noexec，用它自带的隔离安装）
<run> --install-path=<dir> --quiet                          # → <dir>/vendors/fla_npu_transformer
# 装成叠加层（base 不动；包里固定的目录名在这里被改名）
python -m fla_npu.install_opp --overlay --source <dir>/vendors --force
# 查 / 卸
python -m fla_npu.install_opp --status
python -m fla_npu.install_opp --remove-overlay
```

**安装动作的写盘边界（§9.1 V6 实测）**：只写"新目录 `vendors/<overlay_name>/` + `config.ini` 一行"；代码里另有四道保证——`install_overlay()` 只往新目录写、**拒绝**把叠加层装到 base 名下、`remove_overlay()` **拒绝**删 base、overlay 路径中的 `rmtree` 只针对叠加层目录自身。

### 6.5 接口

**CLI（`python -m fla_npu.install_opp`）**

| 参数 | 说明 |
|---|---|
| `--overlay` + `--source <dir\|parent>` | 把源 vendor 装成包内叠加层（`--source` 可给 vendor 目录本身或其只含一个 vendor 的父目录） |
| `--overlay-name <name>` | 叠加层目录名（默认 `fla_npu_overlay_transformer`；禁止覆盖 base 名） |
| `--remove-overlay [--overlay-name <name>]` | 删除叠加层并刷新 `config.ini` |
| `--status` | 打印 `config.ini`、各 vendor 的文件数与算子名 |
| `--install-path <dir> [--force]` | 原有行为：把包内 OPP 拷到外部目录 |

**环境变量**：`FLA_NPU_OVERLAY_VENDOR`（叠加层名列表）｜`FLA_NPU_OP_API_LIB`（**必须指完整库**）｜`FLA_NPU_OP_API_LIBS`（有序库列表，诊断）｜`FLA_NPU_STABLE_ABI`（`ctypes` 强制参考路径）｜`FLA_NPU_DISABLE_PTH`。

**Python API**：`fla_npu.overlay_vendor_names() -> list[str]`｜`fla_npu.load_ascendc_opapi_libraries() -> list[ctypes.CDLL]`（幂等、带缓存）。

### 6.6 运行期解析实例与依赖闭包

**实例**：`chunk_gated_delta_rule_fwd`（复合算子，host 内部在 `aclnn_chunk_gated_delta_rule_fwd.cpp` 调 `l0op::ChunkFwdH` / `l0op::ChunkFwdO`）在子算子 `chunk_fwd_o` 叠加层在位时调用：

| 状态 | 直接调 `chunk_fwd_o` | 复合算子 |
|---|---|---|
| 基线 | — | `digest=a9d2c52a… sum=-0.239569` |
| 装叠加层（子算子 host `scale ×2`） | 官方用例 `test_npu_fwd_o_generalization.py`：**基线 3 PASS → rc=1**，双标杆 `max 0.0537→0.1074`、`mean_re_ratio≈310`（**正好 2 倍**） | **`sum=-0.479137`（正好 ×2）** |
| 卸叠加层 | — | **逐值回退** `-0.239569` |

**解读（机制已由实验定死，勿按"符号插替"理解）**：叠加层替换的产物分布在**不同的层**，而各层的选取规则不同：

| 层（对应 §1.2） | 产物 | 谁决定选哪一份 | 装了叠加层后 |
|---|---|---|---|
| **L1 host aclnn 入口** | `libcust_opapi.so` | **后端**：ctypes 按符号遍历库列表；launcher 按 `FLA_NPU_OP_API_LIBS`/`FLA_NPU_OP_API_LIB` | 新 launcher：overlay；**旧 launcher：base** |
| **L2 算子注册** | `aic-<soc>-ops-info.json` | CANN 按 `ASCEND_CUSTOM_OPP_PATH` 顺序 | **overlay**（与后端无关） |
| **L3/L4 tiling 与 kernel** | `op_tiling/**`（实测 tiling 符号在 `libcust_opmaster_rt2.0.so` 里，`libcust_opapi.so` 里 0 个）、`kernel/<soc>/<op>/*.o` | CANN 按 vendor 顺序 | **overlay**（与后端无关） |

因此：**"入口层"与"tiling/kernel 层"可以来自不同的 vendor** —— 这正是此前一次实验里"入口显示 launcher（`_stable`）却得到 overlay 结果"的原因：那次改的是 **tiling**（`op_host/chunk_fwd_o_tiling_processor.h`），落在 overlay 永远优先的那一层，而旧 launcher 只让 **L1 入口**落回 base，于是出现了**混合状态**。

**两点否定性结论（均有实测）**：① **不存在库内符号插替** —— 装了 overlay 后，`aclnnChunkFwdO`、`aclnnChunkFwdOGetWorkspaceSize` 以及 overlay 导出的 mangled `l0op::ChunkFwdO` 在**全局作用域（`RTLD_DEFAULT`）都取不到**（两库均以 `RTLD_LOCAL` 加载），所以 base 库内部调用不会被 overlay"顶替"；② 复合算子跟随叠加层的现象同样由**分层**解释（其 host 入口在 base 库内链接，但 tiling/kernel 来自 overlay），而不是"运行期重新解析符号"。（§9.1 V20/V21）

**依赖关系事实图谱（决定"要不要编依赖闭包"）**

| 关系 | 证据 | 后果 |
|---|---|---|
| 复合算子 include 子算子**公共 host 头**（`chunk_fwd_o/op_host/op_api/chunk_fwd_o.h`，拿 `l0op::ChunkFwdO` 声明） | `aclnn_chunk_gated_delta_rule_fwd.cpp:9` | **接口/签名**改动 → 依赖方必须一起重编（否则按旧声明编译） |
| 复合算子**自带一份**子算子 kernel 头/实现拷贝（`…/internal/operators/chunk_fwd_o/**`，host tiling 也 include 它） | 6 处 include 全部指向 `internal/…` | 想让**依赖方内嵌那份拷贝**也变 → 必须把依赖方写进 `--ops` |
| `chunk_fwd_o_a5_constants.h` 等 kernel 常量头 | 跨算子引用 **0 处** | 不影响依赖方 |
| 子算子**内部实现**改动 | 上述实例实测 | **不需要**编闭包 |

**共享头改动实验（偏移移动型）**：把 `ChunkFwdOTilingData` **最前面插入一个 `int64_t`**（后续字段偏移 +8 字节），分别：
- **只重编 `chunk_fwd_o`** → 复合算子输出 `a9d2c52a…`（与基线一致）
- **连 `chunk_gated_delta_rule_fwd` 一起重编** → 复合算子输出 `a9d2c52a…`（仍与基线一致）

⇒ 复合算子在运行期**不依赖** standalone 那个结构体的布局（它用自己 `internal/` 下的拷贝），这类"算子内部共享头"改动**不构成跨算子编译闭包**。

**可操作规则**

| 改动类型 | 是否需要把依赖方纳入同一次增量编译 |
|---|---|
| 算子**内部实现**（host 逻辑 / kernel 逻辑） | ❌ 不需要（运行期实时解析到叠加层） |
| 算子**导出的公共头 / 接口签名** | ✅ 需要 |
| 想让**依赖方内嵌的那份拷贝**也跟着变 | ✅ 需要（`--ops=<op>,<dependent>`） |
| 仅本算子内部的 kernel 常量/结构头 | ❌ 不需要（实测偏移移动也不影响依赖方） |

## 7. 备选方案与取舍

### 7.1 方案 A（选定）—— 包内叠加层 vendor + 按符号多库查找

**理由**：① 改动面最小且落在正确抽象层——`_runtime.symbol()` 本就是"按符号逐个库找"，本设计只是把库列表从 1 个扩成 N 个；② 不触碰构建系统（仍走"全量清理 + `--ops` 过滤"）；③ 不碰 run 包安装器，CI 分项 06 断言对象不受影响；④ 实测满足全部目标（§9.1）。

### 7.2 方案 B —— 把叠加层"合并"进 base vendor 树

**做法**：由安装器把叠加层的 kernel 目录、`ops-info` 条目、`binary_info_config` 条目**按算子名 merge** 进 base 同名文件；host 库作为第二个 `.so` 放进 base 的 `op_api/lib`。

**不选**：需要额外实现"合并器"（JSON 按算子名 merge + 目录覆盖 + 备份/回滚），出错面大；而 §1.2 的实测已证明 CANN **按多份索引并集查找**，方案 A 不需要合并。

**何时重考虑**：若某 CANN 版本/工具链要求 `opp/vendors` 下只能有一个 vendor，或出现"同名索引相互覆盖"的行为。

### 7.3 方案 C —— 每次重出整包 / 外部 vendor + 环境变量

- **C1 每次全量重出**：结果一定正确、实现为零，但单次迭代 **884 s**，与 N1 差 14–29 倍。
- **C2 外部 vendor + `ASCEND_CUSTOM_OPP_PATH`**（不动包内）：**kernel 层会生效**（host 只负责 launch，device 二进制按 OPP 树索引），但**host 层改动不会生效**——`fla_npu` 只从包内加载 host 库（§6.3）。要把 host 改动带进来必须重新链接调用方，而调用方就是 `fla_npu` 自身，等于回到包内方案。

**何时重考虑**：C1 在"一次改很多算子"时更简单；C2 可作"只改 kernel"的临时手段（仍需重启进程）。

### 7.4 方案 D —— 对象级增量（复用 `build/**/*.o`）＋ 重链聚合产物

**做法**：保留构建树（去掉每次 `clean`），只重编改动算子的对象，**重新链接** `libcust_opapi.so`、重建 `aic-<soc>-ops-info.json` / `binary_info_config.json`，产出**一份完整 OPP**替换包内那份（不使用叠加层）。

**实测收益与限制**

| 场景 | 实测墙钟 | 说明 |
|---|---|---|
| 冷编译 `build.sh --pkg --ops=chunk_fwd_o`（清树 + configure + 全部目标） | **32 s** | CXX 405 / opc 4 |
| 热树 + 改 1 个 **host** `.cpp` → `package` | **3 s** | 重编 **1** 个 CXX + 重链 + 重出包 → **省 90%** |
| 热树 + 改 1 个 **kernel** 入口 → `ops_transformer_kernel` + `package` | **2 s 但 opc=0**（**无效**） | 当前 CMake 的 kernel 目标**没把 kernel 源文件当依赖**：改源码不触发 opc 重编（需重跑 configure / `generate_compile_cmd`）。→ **对象复用对 kernel 改动目前不可用**（见 R9） |
| 热树 + 不改源码 → 重出包 | **2 s** | 纯重打包成本 |

**不选为默认**：① 产出的仍是**一整份 OPP**（整份替换、无并存、无一键回滚）——安装/回滚这半边问题它不解决；② 对 kernel 改动目前无效；③ 与已固化的"全量清理"契约冲突，需同时改 `build.sh` 与测试断言（**策略变更，需单独评审**）。

**何时采用**：目标从"安全地做单算子迭代"转向"把每次迭代压到秒级"时。最小可行路线：① 去掉 `clean`（保留 configure 缓存）② 修 kernel 目标依赖（或每次重跑 `generate_compile_cmd`）③ 保留聚合重链与索引重建 ④ 与叠加层**并存**：对象复用管"编译"，叠加层管"安装/回滚"。

### 7.5 方案 E —— 把 launcher 适配器也纳入叠加层（未实现，备选）

**动机**：本设计的叠加层只覆盖 vendor 树（`op_api/lib`、`op_tiling`、`kernel`、清单）。而 **launcher 适配器**（`libfla_npu_stable.so`，源码 `torch_custom/fla_npu/csrc/src/stable_<op>.cpp`）是**包内另一个 `.so`**，不在叠加层覆盖范围内。当改动涉及 **aclnn 接口**（新增/删除参数、改输出、改 dtype 约束）时，适配器必须一起改，此时当前通路无法"只编该算子 + 叠加"。

**做法（两种）**：
1. **轻量**：用 `FLA_NPU_STABLE_LIB=<新编的 launcher>` 把 launcher 指到临时产物（`_stable._lib_path()` 优先读该变量），配合 `python csrc/build_stable.py --out <包内路径>` 直接落位；
2. **正式**：包内允许存在多个 `libfla_npu_stable*.so`，叠加层目录里携带自己那份适配器库，加载端按 `overlay → base` 选择，并把"哪份 launcher 生效"纳入 `--status` 审计。

**必须先解决的问题**：**build stamp 校验**（`_stable.py` 的 `_check_build_stamp`）。库把适配器源码哈希编进自身（`fla_npu_stable_source_hash()`），与包内期望值比对；不一致则**整库判不可用 → 所有算子回退 ctypes**（实测：`library 3085f161…` ≠ `package e00df9f5…`）。注释给出两个例外：**不带 stamp 的旧库报 `unknown` 会被接受**；**源码树未生成 stamp 模块则跳过检查**。因此方案 E 需要"库与期望哈希一起更新"的机制（例如把 stamp 也做成叠加层的一部分）。

**何时需要**：改动命中 §12.6.3 表里"必须改适配器"的行（aclnn 接口/输出/可选性变化）。

## 8. 影响面

### 8.1 兼容性

| 维度 | 结论 |
|---|---|
| 默认行为（未装叠加层） | 只返回 base vendor → 加载/安装行为与改动前一致（§9.1 V4 覆盖） |
| 算子接口 / ABI | 不涉及 `def`、`aclnn` 入参、`torch` 入参的增删改 |
| run 包安装器与 CI 分项 06 | 未触碰；`check_scoped_wheel_opp_install()` 断言的告警/状态表不受影响（它仍走 `--full` 覆盖路径）；两者互补 |
| Stable-ABI launcher | `FLA_NPU_OP_API_LIB` 语义被显式固定为"完整库"；装叠加层时后端自动切 ctypes（可用 `FLA_NPU_STABLE_ABI` 覆盖）。代价：叠加层在位期间失去 launcher 的调用开销优势 |
| launcher 独占算子 | `_LAUNCHER_ONLY_OPS`（无 ctypes 参考实现）无法被叠加层覆盖 |
| pip 账本 | 刻意不重写 `RECORD`；`pip install --force-reinstall` 会用 wheel 内容覆盖回 base → 用 `--status` 审计、重装后重跑 `--overlay` |

### 8.2 迁移方案

- **存量用户**：无需动作（wheel 结构、安装方式、默认行为都不变）。
- **要用增量的团队**：① 只编一个算子 `bash build.sh --pkg --soc=<soc> --ops=<op> -j64`；② 取 vendor 树 `<run> --install-path=<dir> --quiet`（注意：run 包不支持 makeself `--noexec --target`）；③ 装成叠加层 `python -m fla_npu.install_opp --overlay --source <dir>/vendors`；④ **重启进程**后调用；`--status` 可随时审计。
- **建议的 CI 集成**：在分项 06 之外新增"叠加层"用例（同一 commit 既跑覆盖安装、也跑叠加安装），断言"其余算子在叠加后仍可用"。

### 8.3 回滚方案

| 层级 | 动作 |
|---|---|
| 运行期 | `python -m fla_npu.install_opp --remove-overlay`（只删叠加层目录 + 刷新 `config.ini`，base 不动） |
| 进程内 | 叠加层不影响已运行进程；重启即回到叠加前状态 |
| 代码期 | 直接回退本分支：3 个 Python 文件，无数据面改动、无 ABI 变化 |
| 兜底 | `config.ini` 被手工改坏时用 `--status` 看出，重跑 `--overlay`/`--remove-overlay` 会重写该文件 |

## 9. 测试与验收

### 9.1 已完成（A5 实机；**全部为正常执行，无异常注入**）

| # | 判据 | 方法 | 结果 |
|---|---|---|---|
| V1 | 叠加生效（**kernel 层**） | kernel 入口 `return;`，只编该算子 | `first8=[1,2,3,4,5,6,7,8] sum=2304` → **`[0,0,…] sum=0`** |
| V2 | 叠加生效（**host 层**） | host tiling `scale *= 2` | 调用**正常成功** → **`[2,4,6,8,10,12,14,16] sum=4608`**（×2） |
| V3 | 叠加生效（**host + kernel 双侧**） | host ×2 + kernel 两处写入点 ×3 | 增量编译 **rc=0 / 63 s / 0 error**；host 库与 kernel `.o` 指纹**都变化**；输出 **`sum=13824`（= 基准 ×6）** |
| V4 | **其余算子可找到且功能正常** | 仓库自带 **10 个算子级用例**，叠加在位/不在位各跑一遍 | **rc 逐项一致**（9 项相同；`test_npu_prepare_wy_repr_bwd_full.py` 单次 >900 s，三态下同样超时、日志逐字节一致） |
| V5 | 输出级对照 | 自建输入（seed 固定）取输出 sha256 | `FastGeluCustom` `6cf13b57da844c10`、`ChunkFwdH` `07854d2fef297a06` **前后完全一致** |
| V6 | **安装写盘边界** | 装/卸叠加层前后对整棵 `opp/` 做逐文件（路径+大小）比对 | base 目录指纹 `32b990df3d3a8ab7`、`libcust_opapi.so` `22cae98457fb`、ops-info `ad09b18d39c2` **全程不变**；除叠加层目录外**唯一被写的既有文件是 `config.ini`**；卸载后整棵 `opp/` 指纹回到安装前 |
| V7 | **运行期实时解析 + 依赖闭包** | 复合算子跟随子算子叠加层；共享头偏移移动实验 | 见 §6.6：复合算子 **×2 且逐值回退**；偏移移动型头改动下"只重编子算子/连依赖方重编"输出**均与基线一致** |
| V8 | 进程级边界 | 同进程"调用→装叠加层→再调用" | 结果不变；**新进程**才生效 |
| V9 | 官方用例三段式 | `test_npu_chunk_local_cumsum.py`（限定算子支持的 spec） | 基线 3 PASS → 装叠加层 **首个用例即 `Mismatched 774/774 (100%)`** → `--remove-overlay` 后与基线**逐字节一致** |
| V10 | 成本 | 整包 vs 单算子 | **884 s（186 次 opc）→ 32–34 s（4 次 opc）**，约 **26×** |
| V11 | 交叉编译（A2/A3） | `--soc=ascend910b` / `ascend910_93` | 各 **9 次 opc / 0 error**，包内为 `config/ascend910b…`、`kernel/ascend910_93…`（**仅编译**） |
| V12 | CANN 多 vendor 语义 | 两个 vendor 同时登记 | **并集**（叠加层在前时 base 的算子仍可用）→ 方案 A 成立 |
| V13 | **直接调用的阳性对照**（补齐 T1） | 用官方用例的真实形状直调 `chunk_fwd_o`（官方入口与 ctypes 参考各一次），叠加层为 host `scale ×2` | 两条路径一致：基线 `digest=9949293fa0ec sum=-3.10511` → 叠加 **`44c2ea1a8321 sum=-6.21021`（×2）** → 卸载**逐值回退**。（此前 161002 的根因是读错 `FwdOCase` 字段顺序：`v_dim=128, chunk_size=64`，与算子无关） |
| V14 | **A2（910B3）真机运行验证** | 在 A2 上**原生构建**（aarch64，CXX 434–443 / opc 18–20 / 0 error，约 3 分钟）后组装独立运行副本；不触碰他人环境 | 基线 `cumsum digest=47054e5fca75 sum=+8320`（**与 A5 完全一致**）；装 host×2 叠加层 → **16640（×2）**；装 kernel×3 叠加层 → **24960（×3）**；对照算子 `fwd_h digest=de2f256064a0` **全程不变**；回滚后回到基线；base 目录指纹 `b14b8a5dcc300704` **不变** |
| V15 | **多叠加层：并存 / 优先级 / 独立卸载** | 同时装 3 个叠加层（其中两个都改 `cumsum`，分别为 ×2 / ×3） | 三个叠加层并存；默认按 `config.ini` 顺序（`a`×2 生效 → 16640）；`FLA_NPU_OVERLAY_VENDOR` 反转顺序后 ×3 生效（24960）；单独卸掉 `a` 后 `cumsum` 回落 base 而**另一个叠加层（fwd_o）仍生效**；全卸回基线 |
| V16 | **多叠加层缺陷修复**（T3 发现） | ① 自定义名叠加层在未设环境变量时**不生效**；② `config.ini` 把 base 排在叠加层之前 | 修复：加载端从 `config.ini` + `fla_npu_overlay_*` 目录**自动发现**叠加层（环境变量仍可覆盖）；安装端把叠加层（含自定义名）**一律排在 base 之前**；`--status` 标注 `[overlay]`/`[overlay?]` 并给出提示。复验：不设环境变量时两个叠加层**都生效**、`config.ini=overlay_a,overlay_b,base` |
| V17 | **"产物变了 ≠ 行为变了"**（A2 踩坑记录） | 只改 kernel 中 4 处 `tiling_->scale` 的其中 1 处（`Muls`，第 712 行）后重建 | kernel `.o` **md5 全部改变**，但**运行结果与 base 完全相同**（该分支对本次输入不执行）→ 改全 4 处后才观测到 ×3。**结论：增量改动的生效必须用行为验证确认，不能只看产物是否变化** |
| V18 | **launcher 与 overlay 的一致性**（上游评审意见） | 评审指出：生效与否只取决于调用方有没有设 `FLA_NPU_STABLE_ABI`，与"是否装了 overlay"不等价；旧 launcher 的 `Runtime::symbol()` 只查 `FLA_NPU_OP_API_LIB` 那**一个**句柄 | 已实现评审建议的**方案 (1)**：`csrc/src/runtime.cpp` 的 `init()` 读取 `FLA_NPU_OP_API_LIBS`（有序，overlay 在前）并 `dlopen` 多个句柄，`symbol()` 按序遍历后再落 CANN；只读单数变量的旧行为保持兼容。编译产出：`libfla_npu_stable.so` 911,456 B → **937,704 B**（含 `FLA_NPU_OP_API_LIBS`）。同时实现**方案 (2) 的兜底**：`_launcher_supports_overlay_order()` 探测 launcher 能力 → 叠加层在位且**未显式指定**时自动切 ctypes；**显式指定 stable 且 launcher 不支持多库**时给出明确告警 |
| V19 | **后端策略矩阵**（A5 实机） | 叠加层在位/不在位 × 环境变量设置 × launcher 新旧 | ①叠加层+未设+旧 launcher → 自动置 `ctypes`（`stable_selected=False`）；②叠加层+**显式 stable**+旧 launcher → 尊重用户但**打印告警**（提示 overlay 会被绕过）；③叠加层+未设+**新 launcher** → 不再降级、launcher 可用（`stable_selected=True`）；④无叠加层 → 无任何告警 |
| V20 | **排除"库内符号插替"** | 装 overlay 后探测全局作用域：`ctypes.CDLL(None)` 取 `aclnnChunkFwdO`、`aclnnChunkFwdOGetWorkspaceSize` 与 overlay 导出的 mangled `_ZN4l0op9ChunkFwdOEPK9aclTensor…`（两个库各自句柄里都能取到） | **全部"全局取不到"** ⇒ 两库都以 `RTLD_LOCAL` 加载、base 库内部调用不会被 overlay 顶替；插替路径**不成立**（此前那次异常另有解释） |
| V21 | **确认"分层归属"** | ①`nm -D` 看 tiling 符号在哪：`libcust_opapi.so` **0 个** Tiling 符号、`libcust_opmaster_rt2.0.so` **183 个**；②运行时 `/proc/self/maps` 看实际加载 | 进程加载的是 **overlay 的 `op_tiling/liboptiling.so`（→ overlay opmaster）** ⇒ 我改的 tiling 落在"CANN 按 vendor 顺序选取"的那一层，与后端无关 ⇒ 解释了"入口 base + 结果 overlay"的混合观测（旧 launcher 只让 **L1 入口**落回 base） |

**开发过程中发现并修复的问题**：Stable-ABI launcher 从 `FLA_NPU_OP_API_LIB` 指向的**单个**库解析全部 aclnn 符号；把它指向只含单算子的叠加层会让其余算子运行期报错（实测 3 个用例由通过转失败）。修复 = 该变量恒指完整库 + 叠加层在位时切 ctypes 后端（§6.2 不变量 1、2）。

### 9.2 未完成 / 交由 CI

| # | 项 | 说明 |
|---|---|---|
| T1 | ~~直接调 `chunk_fwd_o` 的阳性对照~~ | ✅ **已完成**（V13）：根因是读错官方用例的 `FwdOCase` 字段顺序（`v_dim=128, chunk_size=64`），与算子无关 |
| T2 | ~~A2 运行验证~~ | ✅ **已完成**（V14/V15）：在 A2（910B3 真机、aarch64）上原生构建并验证 host/kernel 两层生效、对照算子不变、双叠加层优先级、回滚 |
| T3 | ~~多叠加层行为验证~~ | ✅ **已完成**（V15/V16）：并发现并修复两个缺陷（自动发现、`config.ini` 顺序） |
| T4 | A3（`ascend910_93`）**运行**验证 | 暂无可用的 A3 机器；A3 仅有交叉编译产物（V11） |
| T5 | Example ST / NPU CI 7 分项 | 需在具备 CI 环境的仓库分支上触发 |
| T6 | A2 上复合算子的对照 | A2 的 base 为 5 算子精简集（`ChunkLocalCumsum/ChunkFwdO/ChunkFwdH/ChunkGatedDeltaRuleFwd(+H)`），复合算子 `ChunkGatedDeltaRuleFwd` 在该精简集上报 `161002`；在 A5 的 35 算子 base 上正常（V7 已覆盖）。疑似其内部路径还依赖未编入的配套算子（如 `ChunkScaledDotKkt`/`RecomputeWUFwd`），未进一步定位 |
| T7 | 仓库若干算子用例在当前版本的**既有**失败（spec/形状约束） | 与本改动无关，不属本设计范围 |

## 10. 风险与未决问题

| # | 风险 / 未决 | 影响 | 缓解 |
|---|---|---|---|
| R1 | **launcher 只从单个库解析符号**（`FLA_NPU_OP_API_LIB`） | 指错库会让其余算子运行期报 `symbol not found`（已实际踩到） | 该变量恒指完整库；装叠加层自动切 ctypes（§6.2 不变量 1、2） |
| R2 | **launcher 独占算子**无法被叠加层覆盖 | 这类算子的 host 改动仍需整包 | 文档标注；如需覆盖需补 ctypes 参考实现或改 launcher |
| R3 | 叠加层在位期间失去 launcher 性能优势 | 调用开销上升 | 仅叠加在位时生效；`--remove-overlay` 即恢复 |
| R4 | `pip install --force-reinstall` / 升级 wheel 会覆盖回 base，叠加层可能被"静默回退" | 账本与内容再次脱节 | `--status` 审计；重装后重跑 `--overlay`（或改造成独立子 wheel） |
| R5 | CANN"多 vendor 并集"行为仅在 `9.2.0-beta.2` 实测 | 换版本可能变为"覆盖" | 换 CANN 版本时重跑 V12；若变覆盖则退回方案 B |
| R6 | 叠加层对 pip 不可见（本设计的选择） | 无法用 pip 卸载/审计 | 提供 `--remove-overlay`；后续可评估"独立小 wheel" |
| R7 | 磁盘占用 | 每叠加层约 25 MB（单算子） | 迭代结束即 `--remove-overlay` |
| R8 | 生效需**重启进程** | 无法热更 | 既有约束（`RTLD_NODELETE` + 符号缓存）；已在验收中体现 |
| R9 | **kernel 目标不依赖 kernel 源文件**：热树下改 kernel 入口，`ops_transformer_kernel` **0 次 opc**（实测 2 s 完成） | 决定"对象级增量"目前对 kernel 改动无效；若误以为"去掉 clean 就能增量编 kernel"，会拿到**陈旧 kernel** | 走对象复用路线前先修 CMake 依赖接线（或每次重跑 `generate_compile_cmd`/configure）；在此之前 kernel 改动走叠加层（`--ops` 强制重编） |
| R10 | **"产物变了 ≠ 行为变了"**：A2 实测只改 kernel 中 4 处 `tiling_->scale` 的 1 处，`.o` md5 全部改变但运行结果与 base 相同（该分支对本次输入不执行） | 若用"产物 md5 变化"代替行为验证，会误判增量已生效 | 生效判断一律用**行为验证**（本设计 §9 全部采用输出 digest/sum 对照）；构建产物变化只作为辅助证据 |
| R11 | **自定义名叠加层曾静默不生效**（T3 发现，已修复）：加载端只认默认名或环境变量，安装端把 base 排在自定义名叠加层之前 | 用户执行 `--overlay --overlay-name X` 后看不到任何效果 | 已修：加载端从 `config.ini` + `fla_npu_overlay_*` 目录自动发现；安装端叠加层一律前置；`--status` 标注 active/inactive 并给出提示（V16 复验） |
| R12 | **"混合来源"状态**（V21 定死）：旧 launcher 下 **L1 入口来自 base、L2–L4（tiling/kernel）来自 overlay**；已排除库内插替（V20） | 若改动**同时**涉及 host 入口与 tiling/kernel 的**契约**（如 tiling 结构体布局、参数语义），混合来源可能产生**不自洽**的组合（例如 base 的入口 + overlay 的 tiling），比"整体生效"或"整体不生效"都更难排查 | ①新 launcher 读有序 `FLA_NPU_OP_API_LIBS` → 三层来源一致；②旧 launcher 由策略兜住：未设环境变量→自动 ctypes（三层都走 overlay 优先），显式设非 `ctypes`→**明确告警**；③文档明确"契约类改动建议同时改 L1 与 L3/L4"（见 §6.6 规则表） |

## 11. 里程碑

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 方案设计 + 实现（3 文件）+ 本地验证 | ✅ |
| M2 | A5 实机行为验证（V1–V12） | ✅ |
| M3 | 设计/验证文档 + 开发分支 `feat/opp-delta-overlay` | ✅ |
| M4 | 评审 + 定稿 | ⏳ |
| M5 | 同步到上游 `flashserve/flash-linear-attention-npu` 并触发 NPU CI 7 分项 | ⏳ |
| M6 | 如需：A2/A3 运行验证（CI 提供硬件）、CI 新增"叠加安装"分项、使用指南 | ⏳ 可选 |

## 12. 附录

### 12.1 产物与名词表

#### 12.1.1 文件 vs 符号：一个 `.so` 可以导出很多符号

`.so` 是 ELF 容器，内含一张**动态符号表**（`.dynsym` + `.dynstr`）；**符号 = 表里的一条记录**（名字 + 地址 + 绑定类型），一个文件可导出任意多条（`nm -D` 即打印该表）。单算子包（`chunk_fwd_o`）恰好 **2 条** aclnn 符号：

```
T aclnnChunkFwdOGetWorkspaceSize    ← 第一段：校验参数、算 workspace、组装 executor
T aclnnChunkFwdO                    ← 第二段：真正下发执行
```

即 **aclnn 两段式 API：一个算子通常贡献 2 条符号**；35 算子 → **74 条**（≈2.1/算子，多出来自 `…TensorGetWorkspaceSize`、`…V2…` 等变体）。库里另有大量 `W`（weak）C++ 符号（如 `_Z15DlogRecordInneriiPKcz`），属内部实现。**算子名（`ChunkFwdO`）本身不是符号名**——它是注册表键，由此派生 `aclnn<算子名>` / `aclnn<算子名>GetWorkspaceSize`。

#### 12.1.2 `libcust_opapi.so` 是什么

| 观测（实测） | 值 | 含义 |
|---|---|---|
| `Type` | `DYN (Shared object file)`、x86-64 | 就是一个**动态库** |
| 关键节 | `.dynsym`/`.dynstr`（符号表）、`.text`（代码）、`.rodata`（常量）、`.data`（数据） | **一份编译好的 host 侧 C++ 代码** |
| `NEEDED` | `libnnopbase.so`、`libprofapi.so`、`libopapi_math.so`、libstdc++/libgcc/libc | 站在 **CANN 运行时之上** |
| `SONAME` | **0 条**（`NO_SONAME ON` 生效） | 见 §12.2 |
| **不含** | kernel、算子原型 | device 二进制在 `kernel/<soc>/<op>/*.o`；原型在 `op_proto/` |

**职责**：外部通过它的 aclnn 入口调算子；它负责参数校验、workspace 计算、组装 executor，再由 CANN 运行时把 kernel 下发到设备。

#### 12.1.3 一个 vendor 包的产物名词表（体积取自单算子包实测）

| 名词 | 路径 | 体积 | 是什么 | 谁消费 |
|---|---|---|---|---|
| **host aclnn 实现库** | `op_api/lib/libcust_opapi.so` | 92,304 B | 该算子集的 aclnn C++ 实现 | Python/框架/调用方 |
| **aclnn 头** | `op_api/include/aclnnop/aclnn_chunk_fwd_o.h` 等 | 2,162 / 881 B | C 接口声明 | 编译期 |
| **tiling 库（固定名）** | `op_impl/ai_core/tbe/op_tiling/liboptiling.so` | **42 B 软链** → `lib/linux/x86_64/libcust_opmaster_rt2.0.so`（1,931,480 B） | 形状/属性 → 切分与调度参数 | CANN 运行时 |
| **算子原型库** | `op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so` | 186,384 B | 算子输入/输出/属性定义 | GE / 图模式 |
| **图（ES）插件** | `op_proto/es/lib/linux/x86_64/libes_transformer_cust.so` | 63,560 B | 图模式适配层 | GE |
| **算子清单** | `op_impl/ai_core/tbe/config/<soc>/aic-<soc>-ops-info.json` | 3,762 B（1 算子） | 有哪些算子、什么规格 | CANN（L2） |
| **kernel 索引** | `.../kernel/config/<soc>/binary_info_config.json` | 3,822 B（1 条） | 顶层键即算子名；含 `params.inputs[]` 与选核元数据 | CANN（L3） |
| **kernel 二进制** | `.../kernel/ascend950/chunk_fwd_o/ChunkFwdO_<hash>.o` ×4 | 各 270,784 B | 真正在 NPU 上跑的机器码；一个算子多个变体（`<hash>` 即那组编译参数） | AI Core / AI Vector（L4） |

**一次调用的完整链路**

```text
调用方
 → aclnnChunkFwdOGetWorkspaceSize(...)   [host 库 libcust_opapi.so]
      校验参数 → 读 ops-info 确认算子 → 用 tiling 库(opmaster)算切分 → 算 workspace → 组 executor
 → 调用方按返回值分配 workspace
 → aclnnChunkFwdO(workspace, size, executor, stream)   [同一个 host 库]
      CANN 运行时按 binary_info_config.json 选 kernel
 → 下发 kernel/ascend950/chunk_fwd_o/ChunkFwdO_<hash>.o 到 AI Core/AI Vector
 → 结果写回 out 张量
```

**三个易混库名**：`libcust_opapi.so` = 自定义包的 host 实现库（本设计里被叠加/回落查找的对象）；`libopapi.so` = CANN 自己的总入口（dispatcher，L1 最后一环）；`libopapi_transformer.so` = 内置形态下 transformer 仓的库名。

### 12.2 SONAME 与 dlsym 是两套"名字"机制（常见误解）

**问题**：`libcust_opapi.so` 的 SONAME 是 **0 条**，"Python 进程按名字逐个 dlsym 找 `aclnnXxx`"还成立吗？——**成立，而且正是 0 条 SONAME 才让它可靠成立。**

| 机制 | 作用对象 | 名字是什么 | 谁解析 | SONAME 相关吗 |
|---|---|---|---|---|
| **库级匹配** | 库本身 | **库名**：文件名 / SONAME | 动态链接器 `ld.so`：解析 `NEEDED`、`dlopen("libcust_opapi.so")`（裸名）；并用 SONAME 判断"是否已加载 → 复用句柄" | **相关** |
| **符号级匹配** | 库里的函数 | **符号名**：`aclnnChunkFwdO` 等 | `dlsym(handle, name)`：只在**已打开那个库**的 `.dynsym` 里查 | **无关** |

本设计的两处"按名字找"用的是不同机制：**打开库**用绝对路径 `dlopen`（包内绝对路径由 §6.3 的校验保证）；**取函数**用 `dlsym(句柄, 符号名)`。SONAME 全程不参与。

**实测（同一进程，两份同名文件、都无 SONAME）**

```
C 程序两次 dlopen（绝对路径）：
  handle A=0x5b8b9e340310   handle B=0x5b8b9ea7a110   → 两个独立对象
  dlsym(A,"aclnnSolveTri") = (nil)        ← overlay 库只含 ChunkFwdO
  dlsym(B,"aclnnSolveTri") = 0x71b6e0…    ← base 库有

ctypes：dlsym aclnnChunkFwdO → overlay 有 / base 有，地址不同（0x72cf5e28a520 vs 0x72cf3fdc1250）
        dlsym aclnnSolveTri  → overlay 无 / base 有
```

**为什么 0 条 SONAME 反而是必要条件**：没有 SONAME 时，**库的身份就是它的路径** —— 按两个不同路径 `dlopen` 得到两个独立对象，各查各自符号表；若两份库带**相同 SONAME**，加载器可能把第二次 `dlopen` **复用第一次的句柄**，叠加层就被"静默忽略"（拿到 base 的实现）。这也是 `NO_SONAME ON` 被 `tests/test_wheel_environment.py` 固化的现实意义。

### 12.3 术语

| 术语 | 含义 |
|---|---|
| **叠加层 / overlay vendor** | 与 base vendor 同级、只含若干算子的包内目录，加载优先级高于 base |
| **base vendor** | wheel 内嵌的完整算子包（`fla_npu_transformer`，35 算子） |
| **按符号查找** | `_runtime.symbol()`：遍历库列表逐个 `dlsym`，先命中先用，miss 落下一个库 |
| **聚合产物** | 一个算子集一份：`libcust_opapi.so`、`aic-<soc>-ops-info.json`、`binary_info_config.json` |
| **四条链（L1–L4）** | host 符号 / 算子注册 / kernel 选择 / 实体查找（§1.2） |
| **Stable-ABI launcher** | 不依赖 CPython/libtorch ABI 的启动库 `libfla_npu_stable.so`，从单个库解析全部符号（R1） |
| **aclnn 两段式** | `aclnnXxxGetWorkspaceSize`（第一段）+ `aclnnXxx`（第二段） |
| **SONAME** | 库的二进制身份名；与 `dlsym` 的符号名是两套机制（§12.2） |
| **ctypes 后端 / stable(launcher) 后端** | 调 aclnn 的两种方式：前者纯 Python 手工编组（参考/回退/对照），后者用预编译 C++ 适配器（默认，性能好、不绑 ABI）。详见 §12.6.1 |
| **适配器（adapter）** | `csrc/src/stable_<op>.cpp` 里把 torch 参数编组成 aclnn 调用的一段 C++；编进包内 `libfla_npu_stable.so`，**不属于叠加层覆盖范围**（§7.5） |
| **launcher "携带"某算子** | 该算子在 `csrc/src/stable_ops.cpp` 里注册了适配器 → 走 launcher；不携带则自动回退 ctypes（§12.6.1） |
| **`_LAUNCHER_ONLY_OPS`** | 只由 launcher 承载、ctypes 无参考实现的算子集合；**派生自两个清单的差集**，当前为空（§12.6.5） |
| **build stamp** | launcher 内部记录的**适配器源码哈希**（`fla_npu_stable_source_hash()`）；与包内期望值不一致 → 整库判不可用、全部回退 ctypes（§12.6.4） |
| **`FLA_NPU_OP_API_LIBS`** | 环境变量：有序的 host 库路径列表（overlay 在前）；新 launcher 据此按序遍历多个句柄（§12.6.1） |
| **分层来源（L1 vs L2–L4）** | L1 host 入口由**后端**决定；L2–L4（清单/tiling/kernel）由 **CANN 按 vendor 顺序**决定 —— 两者可以来自不同 vendor，形成"混合状态"（§6.6 / R12） |
| **局部覆盖安装 / scoped overlay** | 上游 CI 分项 06 的既有做法：单算子 run 包 `--full` 覆盖整份 OPP（会 WARNING 其余算子丢失） |

### 12.4 涉及文件与关键位置

| 文件 | 关键点 |
|---|---|
| `torch_custom/fla_npu/fla_npu/__init__.py` | `overlay_vendor_names()`、`_resolve_vendor_dirs()`、`_prepare_embedded_opp()`（反序 prepend、`FLA_NPU_OP_API_LIB`=完整库、按需切 ctypes）、`load_ascendc_opapi_libraries()` |
| `torch_custom/fla_npu/fla_npu/install_opp.py` | `--overlay` / `--remove-overlay` / `--status`；`config.ini` 按优先级写；不重写 `RECORD` |
| `torch_custom/fla_npu/fla_npu_opp_env.py` | `.pth` 启动阶段按 `overlay → base` 预置环境（issue #429） |
| `fla_npu/ops/ascendc/_runtime.py` | `symbol()` 按符号遍历库列表（**未改动**，本设计前提） |
| `fla_npu/ops/ascendc/__init__.py` | 后端选择：`_get_stable_op()` / `_stable_backend_selected()` / `_LAUNCHER_ONLY_OPS` |
| `libfla_npu_stable.so` | 从 `FLA_NPU_OP_API_LIB` 单个库解析全部 aclnn 符号（R1 来源） |
| `scripts/package/ops_transformer/scripts/install.sh` | run 包安装器（"Operator support status…" 告警来源；**未改动**） |
| `ci/run_checks.sh` | 分项 06 `check_scoped_wheel_opp_install()`（**未改动**，与本设计互补） |

### 12.5 复现命令

```bash
source <cann>/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 FLA_NPU_DISABLE_PTH=1
export PYTHONPATH=<隔离运行副本>          # 用 fla_npu 的副本，避免改动原装包
PY=<python>

# 编（单算子）
cd <repo> && bash build.sh --pkg --soc=ascend950 --ops=chunk_local_cumsum -j64
# 取 vendor 树 / 装 / 查 / 卸
bash build_out/fla_npu_linux-x86_64.run --install-path=<dir> --quiet
$PY -m fla_npu.install_opp --overlay --source <dir>/vendors --force
$PY -m fla_npu.install_opp --status
$PY -m fla_npu.install_opp --remove-overlay
```

### 12.6 后端与适配器（stable / ctypes 的分工、代价与迭代通路）

#### 12.6.1 两个后端是什么

先垫一句：**aclnn 是两段式 C 接口**（`aclnnXxxGetWorkspaceSize`：校验 + 算 workspace + 组 executor；`aclnnXxx`：下发），都在 `libcust_opapi.so` 里。两个后端就是"**用什么方式调这两个函数**"：

| 名词 | 是什么 / 在哪 | 干什么 / 特点 |
|---|---|---|
| **ctypes 后端** | 纯 Python：`ops/ascendc/_aclnn_ctypes.py` + `_runtime.py` | 用 `ctypes.CDLL` 打开 `.so` 并按符号取函数指针，在 Python 里**手工构造每个参数**（张量描述符/workspace/stream）。每次调用都要重建描述符 → host 开销大；但不依赖任何二进制 ABI、入参接受面更宽松。定位是**参考实现 / 回退 / 对照** |
| **stable / launcher** | 包内二进制 `fla_npu/libfla_npu_stable.so`；源码 `torch_custom/fla_npu/csrc/src/*.cpp`；编译 `csrc/build_stable.py` | 把"参数封装 + 调 aclnn"用 C++ 预编译，经 `torch.ops.fla_npu_stable.*` 暴露，Python 侧只薄转发。**默认后端**（`FLA_NPU_STABLE_ABI` 不设或非 `ctypes`）。设计目标是**不直接用 CPython 的 C-API**（换 Python 版本不必重编），并用 `csrc/include/torch_stable_abi` 对接 torch；构建时仍链接 `libtorch_cpu/c10/torch` |
| **launcher "携带"某算子** | `csrc/src/stable_ops.cpp` 里 `m.def(kSchema_…)` 逐条注册（+ `csrc/src/stable_<op>.cpp` 一份手写适配器） | 携带 = 有手写适配器 → 走 launcher（快）；**不携带 = 自动回退 ctypes**（结果相同，只是慢） |
| **`FLA_NPU_OP_API_LIBS`** | 环境变量（`os.pathsep` 分隔的 `.so` 路径列表，前者优先） | 由 `import fla_npu` 写入 `[overlay, base]`；**新** launcher 在 `init()` 里逐个 `dlopen` 并按序遍历 `dlsym`（旧 launcher 不认识它 → 需要能力探测 + 降级/告警，见 V18/V19） |

#### 12.6.2 stable 的收益与代价

| | 内容 |
|---|---|
| 收益 | ① host 开销低（省掉每次调用重建描述符，文档称 ctypes 的 host 开销是其"数倍"）；② **不绑定 CPython / libtorch 的 C++ ABI** → 一个 wheel 可跨 Python/torch 版本 |
| 代价 | ① 覆盖面靠**手写**适配器（当前 36 个算子 36 份）；② **build stamp 强耦合**——库与包的适配器源码不同源即**整库判不可用**（实测踩到：`library 3085f161…` ≠ `package e00df9f5…`）；③ 依赖 torch 的少量稳定符号（`aoti_torch_*`），版本不符即不可用；④ 参数语义要在 C++ 里再复述一遍（可选张量 / `int[]?` / 输出顺序 / in-place 契约），易与 ctypes 侧不一致；⑤ 出错堆栈落在 `.so` 里，调试比纯 Python 难；⑥ 引出"符号来源"问题（已修，见 §7.5 / R12） |

#### 12.6.3 适配器 vs kernel / host：改什么要连带改什么

| 层 | 产物 | 在叠加层覆盖范围内吗 |
|---|---|---|
| **适配器**（host 参数编组） | `libfla_npu_stable.so`（**包内**） | ❌（见 §7.5 方案 E） |
| **host aclnn 实现 / tiling** | `libcust_opapi.so`、`op_tiling/**`（vendor 树） | ✅ |
| **device kernel** | `kernel/<soc>/<op>/*.o`（vendor 树） | ✅ |
| 算子定义 / 清单 / 索引 | `ops-info.json`、`binary_info_config.json`（vendor 树） | ✅ |

| 算子侧改动 | 要改适配器吗 | 还要动什么 |
|---|---|---|
| 只改 **kernel**（数值/性能，接口不变） | ❌ | 重编算子包（`--ops`）即可；launcher 不用动 |
| 只改 **host 实现 / tiling**（aclnn 接口不变） | ❌ | 同上 |
| **aclnn 接口签名**变化（增删/改序/改类型/改 attr） | ✅ | schema（`npu_custom.yaml`）+ ctypes wrapper + 测试 |
| 输出个数/顺序/可选性变化、dtype/layout 约束变化 | ✅ | 同上 |
| 新增 / 重命名算子 | ✅ | 三处清单（`_ASCENDC_OPS`、`ASCENDC_CTYPES_OPS`、`csrc/src/stable_ops.cpp`）+ 新 `stable_<op>.cpp` |
| 改 `csrc/**`（即使只是注释/格式） | ✅ **必须重编 launcher** | 否则 stamp 变化 → 整库判不可用 |

（关键：适配器是按**符号名** `dlsym` 调 aclnn 的，所以**只要 aclnn 接口不变，改 kernel/host 都不需要碰适配器** —— 这正是单算子增量最常走的路径。）

#### 12.6.4 适配器迭代通路（现状）

| 通道 | 用法 | 注意 |
|---|---|---|
| 指向临时产物 | `FLA_NPU_STABLE_LIB=<新编的 .so>`（`_stable._lib_path()` 优先读该变量） | 会撞 build stamp 校验 |
| 直接落位 | `python csrc/build_stable.py --out <包内路径>` | 官方错误提示给出的做法 |
| 两个例外 | 不带 stamp 的旧库报 `unknown` **会被接受**；源码树未生成 stamp 模块则**跳过检查** | 决定"能否只换适配器" |

#### 12.6.5 `_LAUNCHER_ONLY_OPS` 的设计取向与风险

定义处自带注释：**派生、不手写**（`= _ASCENDC_OPS - ASCENDC_CTYPES_OPS`），取向是"**新算子不再要求提供 ctypes 适配器**"——用"少维护一份实现"换"对 launcher 的强依赖"。**实测当前集合为空**（`_ASCENDC_OPS` 35 个、ctypes 侧 35 个 `npu_*` 函数），所以"没有兜底"目前只是**预期演化方向**。代价：一旦某算子成为 launcher-only 而 launcher 不可用（stamp 不符 / 缺符号 / wheel 以 `FLA_NPU_BUILD_STABLE_ABI=0` 编出），调用直接抛 `AttributeError`。**建议（未实施）**：CI 增加"launcher 必须可用"的断言，并让 stamp 校验失败更可见。

#### 12.6.6 生成适配器"缺的语义"在哪一层（以 `chunk_fwd_o` 为例）

| 需要的语义 | 在哪一层 | 实测 |
|---|---|---|
| 对外签名（两段式入口、形参顺序/类型） | `op_host/op_api/aclnn_chunk_fwd_o.h` | **只有原型**，没有可选性/dtype |
| **必选/可选、dtype、format、value-depend、输出** | **`op_host/chunk_fwd_o_def.cpp`** | `Input("q").ParamType(REQUIRED).DataType({DT_BF16, DT_FLOAT16,…})`；**`Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)`**；`Output("o")` |
| 属性与默认值 | 同上 `def.cpp` | `Attr("scale").AttrType(REQUIRED).Float(1.0)` |
| torch 侧 schema | `torch_custom/fla_npu/npu_custom.yaml` 的 `func:` 行 | `npu_chunk_fwd_o(Tensor q, …, Tensor? g=None, …, int? chunk_size=None, …, str output_layout="BNSD") -> Tensor` |
| 约束/校验 | `op_host/chunk_fwd_o_tiling*.cpp/.h` | chunk_size、layout 枚举等 |
| **编组细节**（谁 `tensor()` / 谁 `int_array()` / 标量类型） | `csrc/src/stable_chunk_fwd_o.cpp` | `:94 FLA_STABLE_EXEC("aclnnChunkFwdO", …, tensor(meta_of(q)), …, int_array(chunk_indices), scalar(scale), scalar(chunk_size), …)` |

**结论**：`def.cpp`（可选性/dtype/输出）+ `npu_custom.yaml`（torch schema）+ `aclnn_<op>.h`（对外原型）**三处交叉即可生成适配器骨架**；仍需人工确认的是"值依赖参数如何编组、`Format` 与实际内存排布的对应、attr 与 tiling 校验的耦合"——这三类目前以手写 ctypes wrapper（`_aclnn_ctypes.py` 同名函数，等价于一份人写的参考实现）与 tiling 代码为准，可作为**第四来源**交叉校验。

### 12.7 关键实测数字速查

| 项 | 数值 |
|---|---|
| 整包（A5，35 算子） | 105,811,197 B；884 s；CXX 641 / opc 187 |
| 单算子叠加包（A5） | 1,391,858 B；30–63 s；CXX 405 / opc 4；339–346 文件 |
| A2/A3 交叉编译（A5 上产出） | 2,391,440 B / 2,391,458 B；各 9 次 opc；0 error（**仅编译**） |
| A2 **原生**构建（910B3，aarch64） | base（5 算子）8,529,133 B / CXX 443 / opc 20 / 0 error / 约 3 分钟；单算子叠加包 2,362,127 B（host）与 2,386,999 B（kernel）/ opc 9 |
| cumsum 基线 digest | A5 与 A2 **完全一致**：`47054e5fca75 sum=+8320` |
| host+kernel 双侧改动（A5） | 输出 `sum=2304 → 13824`（×6） |
| 单算子叠加层（A5 直调 `chunk_fwd_o`） | 基线 `sum=-3.10511` → ×2 `-6.21021` → 回退 |
| A2 叠加层效果 | host×2 → `sum=8320→16640`；kernel×3 → `24960`；对照算子 `de2f256064a0` 不变 |
| 其余算子回归 | 10 用例 9 项 rc 一致；指纹 `6cf13b57…` / `07854d2f…` 不变 |
| 安装写盘边界 | base 指纹 `32b990df3d3a8ab7`（A5）/ `b14b8a5dcc300704`（A2）不变；仅 `config.ini` 一行被写 |
| 对象复用（方案 D，参考） | 热树改 host `.cpp` → 3 s；改 kernel → 0 次 opc（R9） |
