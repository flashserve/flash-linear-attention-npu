# FLA NPU 算子包「单算子叠加层」增量设计

- **作者**：zzy1205
- **状态**：已适配主干 Stable-only；历史 A5/A2 证据见[修复验证记录](PR878最终检视修复验证.md)。本次新增 launcher 能力门禁的离线验证与待补硬件验证见 §10，不继承旧双后端验收结论。
- **适用范围**：`flash-linear-attention-npu` wheel 内嵌 OPP 的单算子或少量算子迭代
- **过程归档**：[《OPP 叠加层设计过程记录》](OPP叠加层增量设计-过程记录.md)
- **兼容性准则**：[《FLA 构建兼容性验证准则》](FLA构建兼容性验证准则.md)

> 本文是规范性设计文档，只描述最终架构、接口、约束和验收结论。实验环境、失败排查、阶段性判断、完整验证矩阵及未采用的细碎方案均移入过程记录。两篇文档冲突时，以本文和代码为准。

## 1. 背景与问题

`fla_npu` 将自定义 OPP 内嵌在 wheel 中：

```text
site-packages/fla_npu/opp/vendors/fla_npu_transformer/
```

一份 vendor 同时包含 host 入口库、算子清单、kernel 索引、tiling/proto 和 device kernel。现有全量包适合作为发布基线，但不适合单算子迭代：用只含一个算子的包覆盖完整 vendor 后，未包含的算子会同时失去 host 符号、算子清单条目和 kernel 实体。

问题不在于构建命令能否过滤算子，而在于**过滤后的聚合产物如何与完整基线共存**。设计必须同时满足：

1. 改动算子优先使用增量实现；
2. 增量未提供的算子自动回落到完整 base；
3. L1 host aclnn 入口与 CANN 侧 L2 自定义算子发现/登记、L3-L4 tiling/kernel 查找遵循一致的 `overlay -> base` 来源优先级；
4. 安装和卸载不修改 base vendor；
5. 进程能明确知道实际加载了哪些来源。

### 1.1 运行期查找模型

```mermaid
flowchart LR
    P[Python Stable wrapper] --> S[Stable launcher]
    S --> L1[L1: overlay → base → CANN host]
    L1 --> C[CANN runtime]
    C --> L2[L2: overlay → base → built-in 算子清单]
    C --> L3[L3/L4: 同顺序 tiling 与 kernel]
```

一次自定义算子调用涉及四条查找链：

| 层 | 查找对象 | 产物 | 选择方 |
|---|---|---|---|
| L1 | host aclnn 符号 | `op_api/lib/libcust_opapi.so` | Stable-ABI launcher（wheel 内的 C++ aclnn 调用库 `libfla_npu_stable.so`） |
| L2 | 自定义算子发现与登记 | `aic-<soc>-ops-info.json` | CANN |
| L3 | kernel 变体 | `binary_info_config.json` | CANN |
| L4 | tiling/proto/kernel 实体 | `op_tiling/**`、`op_proto/**`、`kernel/<soc>/<op>/*.o` | CANN |

这里的“发现与登记”不是 Python/torch 接口注册。CANN 初始化自定义 OPP 时，会按 vendor 优先级读取 `aic-<soc>-ops-info.json`，确认某个自定义算子存在并把它的实现元数据纳入运行时查找；后续调用才能继续进入该算子的 tiling 与 kernel 变体选择。

这里有两个关键事实：

- **查找按符号或算子逐项发生**：当前来源没有该项时，可以继续查下一个来源。
- **产物按算子集聚合**：一份 `libcust_opapi.so` 或索引文件承载多个算子，不能把单算子聚合产物当作完整包替换。

## 2. 方向选择

设计阶段只保留一个需要长期理解的方向选择：

| 方向 | 核心做法 | 主要代价 |
|---|---|---|
| 单一来源 | 覆盖或合并进 base，使运行期始终只看到一棵 vendor 树 | 覆盖会丢算子；合并需要修改 base、合并聚合索引并处理回滚 |
| **有序多来源 + 逐项回落** | 保留完整 base，把增量作为更高优先级 vendor；每层按相同顺序逐项查找 | 需要统一 L1 与 L2-L4 的来源顺序，并管理旧 launcher 兼容 |

本设计选择第二个方向。原因是它直接利用现有运行时能力：

- `dlsym` 可以按句柄顺序逐符号查找；
- CANN 可以按 vendor 顺序逐算子查找算子清单、tiling 和 kernel；
- 单算子增量只覆盖自己携带的项，其他项自然回落 base；
- base 保持只读，安装、审计和回滚边界清晰。

这个选择形成全文的核心不变量：

> **L1 与 L2-L4 必须使用同一个有序来源列表：`overlay... -> base -> CANN`；每一项由第一个提供它的来源承载。**

## 3. 目标与非目标

### 3.1 目标

- 支持只构建一个或少量算子，并安装为包内 overlay vendor。
- overlay 中存在的算子覆盖 base，不存在的算子自动回落 base。
- 唯一执行后端 Stable-ABI launcher 必须按与 CANN 一致的 host 库顺序查找。
- CANN 侧用于确认“自定义算子存在”的算子清单，以及后续 tiling、kernel 的查找顺序，与 L1 host aclnn 库保持同一 `overlay -> base` 优先级。
- 支持多个 overlay，顺序明确、可审计、可单独卸载。
- base vendor 在 overlay 安装和卸载过程中保持不变。
- 为旧版 launcher、overlay 启用规则和已有错误路径定义可验证的兼容行为。

| 兼容场景 | 业务含义与代码出处 | 确定行为 |
|---|---|---|
| 旧版 Stable-ABI launcher | wheel 内的高性能 C++ aclnn 调用库 `libfla_npu_stable.so`；旧版 `Runtime::symbol()` 只查 `FLA_NPU_OP_API_LIB` 指向的一个 custom 库 | 安装/import 先验证公共 ABI；首次加载实际 launcher 时若缺少有序多库能力导出或版本小于 1，硬失败，不解析或执行算子 |
| `FLA_NPU_OVERLAY_VENDOR` | `fla_npu/__init__.py` 与 `.pth` 启动钩子读取的环境变量，用于选择启用哪些 overlay 及其优先级 | 未设置时按 `config.ini`/目录自动发现；显式名称列表按给定顺序；显式空值时禁用全部 overlay，只使用 base |
| 已有错误 OPP 路径 | shell 可能已把 base 放在 overlay 前面，或遗留 `<pkg>/opp`、`vendor/op_api/lib` 条目 | 自动清理当前包的旧条目并归一化为 `overlay -> base -> external`；实际发生修正时告警一次 |

### 3.2 非目标

- 不提供进程内热更新；安装或卸载后必须启动新进程。
- 不把 CMake 对象缓存改造成通用对象级增量构建系统。
- 不放宽 host 库只能从当前 wheel 包内加载的边界。
- 不修改算子的 `def`、aclnn 或 torch schema。
- 不改变既有 run 包的 `--full` 覆盖安装语义。

## 4. 总体架构

下列目录与 §4.1 调用链定义当前 Stable-only 架构；历史双后端图片仅保留在过程记录中。

```text
site-packages/fla_npu/
├── libfla_npu_stable.so               # wheel 级 launcher，不属于任何 vendor
└── opp/vendors/
    ├── config.ini
    ├── fla_npu_overlay_transformer/   # 本 PR 新增的默认 overlay vendor
    │   ├── op_api/lib/libcust_opapi.so
    │   ├── abi_manifest.json          # 导出契约与 host 库/公共头 SHA256
    │   ├── op_impl/.../config/<soc>/aic-<soc>-ops-info.json
    │   ├── op_impl/.../kernel/<soc>/<op>/*.o
    │   ├── op_impl/.../op_tiling/
    │   └── op_proto/
    └── fla_npu_transformer/           # PR 前已有的完整 base，名称保持不变
```

`fla_npu_transformer` 不是本 PR 新起的名称：既有构建流程使用逻辑 vendor 名 `fla_npu`，打包时追加固定 `_transformer` 后缀形成这个物理目录。本设计不重命名 base，只新增带 `overlay` 标识的 `fla_npu_overlay_transformer`。

`libfla_npu_stable.so` 由 wheel 构建流程生成并放在 `fla_npu/` 包根目录，是所有 vendor 共用的 Stable-ABI 调用库。单算子 overlay 只携带 OPP vendor 产物，不重复携带 launcher。

`config.ini` 的优先级示例：

```ini
load_priority=fla_npu_overlay_transformer,fla_npu_transformer
```

### 4.1 wheel launcher 的内部组成与调用关系

`wheel launcher` 指 wheel 中的共享库文件 `fla_npu/libfla_npu_stable.so`；`C++ Runtime::symbol()` 不是另一个产物，而是编译在这个共享库内部的成员函数。launcher 的逻辑组成如下：

```text
fla_npu/libfla_npu_stable.so              # wheel launcher 产物
├── 每个算子的 C++ adapter                # stable_<op>.cpp 编入同一 .so
│   └── torch 参数 -> aclnn 参数/描述符
└── fla_npu_stable::Runtime               # runtime.cpp 编入同一 .so
    ├── Runtime::init()
    │   └── 按 FLA_NPU_OP_API_LIBS dlopen overlay/base，再打开 CANN libopapi.so
    └── Runtime::symbol(name)
        └── 按 overlay -> base -> CANN dlsym，并缓存首个命中的函数地址
```

三者职责不同：

| 组件 | 职责 | 不负责什么 |
|---|---|---|
| wheel launcher：`libfla_npu_stable.so` | 承载所有 C++ adapter 和 `Runtime`，通过 torch dispatcher 暴露稳定调用入口 | 不携带某个 overlay 的 host/tiling/kernel 实现 |
| 算子 C++ adapter | 把 torch Tensor、可选参数和标量编组成对应 `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 调用 | 不决定符号来自 overlay 还是 base |
| `Runtime::symbol()` | 根据有序库句柄查找并缓存 `aclnnXxx*` 函数地址 | 不实现算子，也不直接选择 tiling 或 device kernel |
| vendor `libcust_opapi.so` | 提供实际的 custom `aclnnXxx*` host 实现 | 不包含 wheel launcher |

Stable-ABI 路径的一次完整调用为：

```text
Python npu_xxx(...)
  -> 唯一 Stable wrapper
  -> 首次 load：严格 build stamp + overlay 有序多库版本门禁
  -> torch.ops.fla_npu_stable.npu_xxx
  -> libfla_npu_stable.so 中该算子的 C++ adapter
  -> Runtime::symbol("aclnnXxxGetWorkspaceSize" / "aclnnXxx")
  -> overlay/base 的 libcust_opapi.so，最后回落 CANN libopapi.so
  -> 命中的 aclnn host 实现发起算子执行
  -> CANN 按同一 vendor 优先级完成 L2 算子发现、L3 kernel 变体选择和 L4 tiling/kernel 执行
```

因此，launcher 决定“如何从 torch 调用 aclnn”，adapter 决定“参数如何编组”，`Runtime::symbol()` 决定“实际调用哪一份 aclnn host 符号”；CANN 再独立完成 tiling 和 kernel 选择。`Runtime::symbol()` 会缓存函数地址，这也是安装/卸载 overlay 后必须启动新进程的原因之一。

### 4.2 行为相关产物清单

| 产物 | 所在位置 | 作用与归属 |
|---|---|---|
| `.pth` 启动钩子 | `site-packages/fla_npu_opp_env.pth` + `.py` | 在 CANN 初始化前选择并排列 vendor |
| Python 分发与 wrapper | `fla_npu/__init__.py`、`fla_npu/ops/ascendc/*.py` | 解析来源、组织 Stable 调用参数；顶层 import 不导入 Torch |
| launcher build-stamp | `fla_npu/ops/ascendc/_stable_hash.py` | 确认 launcher 二进制与同 wheel adapter 源码匹配 |
| Stable-ABI launcher | `fla_npu/libfla_npu_stable.so` | wheel 级 L1 C++ 调用入口，内含算子 adapter 与 `Runtime::symbol()`，所有 vendor 共用 |
| vendor 优先级 | `opp/vendors/config.ini` | 记录 overlay 到 base 的默认顺序 |
| shell 环境脚本 | `<vendor>/bin/set_env.bash` | source 时归一化 `ASCEND_CUSTOM_OPP_PATH` |
| public aclnn 头 | `<vendor>/op_api/include/aclnnop/*.h` | 编译、安装和运行期可检查的 L1 ABI 契约 |
| ABI 清单 | `<vendor>/abi_manifest.json` | 打包时验证导出覆盖；安装/import 时核验库与头文件完整性 |
| custom host 库 | `<vendor>/op_api/lib/libcust_opapi.so` | L1 aclnn 实现 |
| 算子发现清单 | `<vendor>/op_impl/.../config/<soc>/aic-*-ops-info.json` | L2：CANN 确认自定义算子存在 |
| 实现源码/JIT 支撑 | `<vendor>/op_impl/.../<vendor>_impl/{dynamic,ascendc}/**` | 随包携带的实现源码、公共头和动态/JIT 支撑文件；不单独决定 overlay 顺序 |
| kernel 索引 | `<vendor>/op_impl/.../kernel/config/<soc>/binary_info_config.json` | L3：选择 kernel 变体 |
| tiling 库 | `<vendor>/op_impl/.../op_tiling/lib/liboptiling.so` 等 | host 侧生成 tiling data |
| device kernel 与旁车元数据 | `<vendor>/op_impl/.../kernel/<soc>/<op>/{*.o,*.json}` | L4：NPU 执行实体及其二进制描述 |
| proto/ES | `<vendor>/op_proto/**`、`libcust_opsproto_rt2.0.so`、`libes*.so` | 算子原型、proto 与表达式相关产物 |
| CANN 兜底 | CANN 安装目录中的 `libopapi.so` 与 `opp/built-in` | wheel 外部的最后一级 L1、L2-L4 回落来源 |

架构图独立展示上述会参与来源选择、ABI 检查或实际执行的产物；README、`version.info`、pip 元数据等不改变查找决策的文件不展开。

### 4.3 各层的有序来源

| 层 | 最终顺序 | miss 行为 |
|---|---|---|
| Stable-ABI host 符号 | `FLA_NPU_OP_API_LIBS` 中的 `overlay_lib... -> base_lib -> libopapi.so` | 继续下一个句柄 |
| CANN 自定义算子清单/tiling/kernel | `ASCEND_CUSTOM_OPP_PATH` 中的 `overlay_vendor... -> base_vendor`，最后由 CANN built-in 兜底 | 继续下一个 vendor |

同名算子由最高优先级 overlay 覆盖；overlay 未携带的算子在 base 命中。多个 overlay 时，靠前者优先。

### 4.4 关键数据与环境变量

| 名称 | 语义 |
|---|---|
| `vendor_dirs` | 当前 wheel 内的有序 vendor 列表：`[overlay..., base]` |
| `FLA_NPU_OVERLAY_VENDOR` | 可选的 `os.pathsep` 分隔 overlay 名称列表；设置后覆盖自动发现顺序 |
| `FLA_NPU_OP_API_LIBS` | 当前 launcher 使用的有序 host 库列表：`overlay... -> base` |
| `FLA_NPU_OP_API_LIB` | 单 vendor 输入，始终指向完整 base 库；启用 overlay 时以复数列表为准 |
| `ASCEND_CUSTOM_OPP_PATH` | CANN 自定义 OPP vendor 目录列表，只登记 vendor 目录 |
| `FLA_NPU_OPP_PATH` | 当前 wheel 的 OPP root，供诊断使用，不参与 `ASCEND_CUSTOM_OPP_PATH` 查找 |
| `FLA_NPU_STABLE_ABI` | 兼容开关；未设置/空值/`stable` 及启用别名 `1/true/yes/on` 使用 Stable，`ctypes` 和其他无效值直接报错 |
| `FLA_NPU_DISABLE_PTH` | 禁用解释器启动阶段的 `.pth` 环境预置 |

`FLA_NPU_OVERLAY_VENDOR` 的状态必须严格区分：

| 状态 | 行为 |
|---|---|
| 未设置 | 从 `config.ini` 和包内合法的 `fla_npu_overlay_*` 目录自动发现 |
| 显式名称列表 | 仅使用列表中存在且合法的 overlay，保持用户给定顺序 |
| 显式空字符串、纯空白或纯分隔符 | 禁用所有 overlay，只使用 base |

`.pth` 启动钩子、`import fla_npu` 和安装工具必须遵守同一套 `FLA_NPU_OVERLAY_VENDOR` 选择规则。

### 4.5 `ASCEND_CUSTOM_OPP_PATH` 语义与归一化

`ASCEND_CUSTOM_OPP_PATH` 的每个条目是一个 **vendor 目录**。以下内容不应写入该变量：

- 当前 wheel 的 `<pkg>/opp` root；
- `vendor/op_api/lib`；
- CANN 的 `opp/built-in`。

当前 wheel 只接管自己 `<pkg>/opp/` 下的路径。写入前先移除当前包已有的 root、vendor 和 `op_api/lib` 条目，再按 `overlay... -> base` 一次性放到最前面。

**external 条目**指进程原有变量中所有不位于当前 wheel `<pkg>/opp/` 下的其他自定义 vendor，例如其他算子包的 `/opt/other_ops/vendor_a`。归一化不删除这些条目，也不改变它们彼此的相对顺序：

```text
归一化前：external_a -> base -> external_b -> overlay
归一化后：overlay -> base -> external_a -> external_b
```

CANN built-in 不属于 external；它不写入 `ASCEND_CUSTOM_OPP_PATH`，仍由 CANN 自身查找链在 custom vendor 之后兜底。

当已有包内条目确实被重排或清理时，启动钩子打印一次 `RuntimeWarning`；干净环境首次登记不告警。生成的 `set_env.bash` 同样读取 `config.ini` 并重建完整顺序，因此 source base/overlay 的先后不会改变结果。

### 4.6 包内来源边界

overlay 与 base 都必须位于当前 wheel 的 `opp/vendors/` 下。运行时对 vendor 和 `libcust_opapi.so` 分别执行 `resolve()` 与 `relative_to(package_dir)` 校验，因此包内软链指向包外也会被拒绝。

这个边界保证：

- host 本地代码来源可审计，不随外部 CANN 目录漂移；
- CANN 的 vendor 发现与 Python 的 host 库加载使用同一份包内 OPP；
- 外部 vendor 不能仅靠环境变量替换当前 wheel 的 host 实现。

## 5. 运行时设计

### 5.1 启动阶段

随 wheel 安装的 `fla_npu_opp_env.pth` 在 site 初始化阶段执行，早于用户代码和通常的 `import torch_npu`：

```text
Python site 初始化
  -> fla_npu_opp_env.py
       -> 解析 FLA_NPU_OVERLAY_VENDOR
       -> 得到 [overlay..., base]
       -> 归一化 ASCEND_CUSTOM_OPP_PATH
       -> 设置 FLA_NPU_OPP_PATH
```

提前设置路径是必要条件：CANN 初始化后再修改 `ASCEND_CUSTOM_OPP_PATH`，已经载入的自定义算子清单和 tiling/kernel 索引不会自动刷新。

### 5.2 `import fla_npu`

```text
import fla_npu
  -> 运行版本与产品 guard
  -> 检查启动阶段是否已预置 OPP 路径
  -> _resolve_vendor_dirs()               # [overlay..., base]
  -> 再次归一化 ASCEND_CUSTOM_OPP_PATH    # 幂等
  -> 设置 FLA_NPU_OP_API_LIB(S)
  -> 加载 libopapi.so 与各 custom libcust_opapi.so
  -> 缓存 [overlay_lib..., base_lib, cann_lib]
```

顶层 import 只完成路径、公共 ABI 与 host 库准备，不导入 Torch。首次 `_stable.load()` 加载 launcher 并通过 build stamp 与能力门禁后，C++ runtime 读取 `FLA_NPU_OP_API_LIBS`，逐个 `dlopen` 并按相同顺序 `dlsym`；custom 全部 miss 后才落 CANN。

### 5.3 overlay 状态与唯一 Stable 后端

| 实际加载的 Stable launcher | 未启用 overlay | 启用 overlay |
|---|---|---|
| 导出有序多库版本 `>= 1` | C++ `Runtime::symbol()`：`base -> CANN` | `overlay... -> base -> CANN` |
| 缺失导出或版本 `< 1` | 仍须通过加载、严格 build stamp、算子注册检查 | 首次 `load()` 硬失败，不解析算子，不执行任何 kernel |

“启用 overlay”指 `FLA_NPU_OP_API_LIBS` 中有多个非空库路径。目录存在但 `FLA_NPU_OVERLAY_VENDOR=""` 时仅选择 base。CANN 的 L2-L4 同样按启用后的 vendor 列表查找。没有第二执行后端，也没有告警后允许 base host 与 overlay tiling/kernel 混用的分支。

### 5.4 Launcher 能力与失败时机

1. **公共 ABI**：构建期比较 adapter 与 OPP 头；安装/import 比较 overlay/base 的清单、哈希、公共 aclnn 原型。失败即拒绝。
2. **实际二进制**：首次 `_stable.load()`（通常由第一次算子调用触发）按 `FLA_NPU_STABLE_LIB` 或包内路径调用 `torch.ops.load_library()`，再严格验证 `_stable_hash.py` 与库的 source hash。
3. **有序查找能力**：启用 overlay 时，对上一步的同一实际路径调用 `ctypes.CDLL(path)`，读取显式 C 导出 `int32_t fla_npu_stable_overlay_order_version()`；当前实现返回 `1`。导出不存在、探测异常或版本 `< 1` 均抛错。
4. **注册与执行**：只有以上校验通过后才记录成功加载状态、解析 `_op(name)` 并允许 adapter 调用。缺失注册同样硬失败。

```text
安装 / import
  -> 校验 overlay/base ACLNN ABI 与产物完整性
  -> 准备 overlay -> base 的 OPP / host 库顺序（不导入 Torch）

首次 _stable.load() / 算子调用
  -> 加载实际 launcher（包含 FLA_NPU_STABLE_LIB 覆盖）
  -> 严格 build stamp
  -> 多库启用？
       ├─ 否：继续 base-only 注册检查
       └─ 是：实际库导出 overlay_order_version >= 1？
                ├─ 否：硬失败，禁止算子解析/执行
                └─ 是：注册检查 -> overlay -> base -> CANN
```

**安装成功仅证明 ACLNN ABI/产物校验通过，不证明实际 launcher 的多库能力已经通过。** 能力门禁位于首次 load，不能为了提前探测而在 `import fla_npu` 时导入 Torch、改变 OPP 在 CANN 初始化前准备的顺序。这里使用标准库 ctypes 读取一个 C 能力函数，不恢复 ctypes 算子执行后端。

公共 aclnn 原型一致不能证明 base host 与 overlay 内部 tiling data 格式兼容，因此所有不具备有序多库能力的 launcher 都被阻止；不保留强制 stable 告警继续、自动 ctypes 或二进制字符串猜测路径。无法执行时仍可用不导入运行时的 `python -m fla_npu_opp_manage --status` / `--remove-overlay --overlay-name <name>` 恢复，随后启动新进程。

接口不变的 host/tiling/kernel 修改可复用**已经具备上述能力且 stamp 匹配**的 wheel launcher。aclnn/schema/adapter 契约改变或新增公共算子必须同步更新完整 wheel 的 wrapper、schema、adapter、算子清单和测试。overlay 本身不携带 launcher；`FLA_NPU_STABLE_LIB` 覆盖的库也必须满足同样的 stamp 与能力要求。

## 6. 安装、卸载与审计

### 6.1 增量工作流

```bash
# 1. 只构建受影响算子
bash build.sh --pkg --soc=<soc> --ops=<op>[,<dependent>...] \
  -j64

# 2. 将 run 包解到隔离目录
build_out/fla_npu_*.run --install-path=<dir> --quiet

# 3. 安装为当前 wheel 内的 overlay
python -m fla_npu_opp_manage \
  --overlay \
  --source <dir>/vendors \
  --force

# 4. 审计或回滚
python -m fla_npu_opp_manage --status
python -m fla_npu_opp_manage --remove-overlay
```

安装或卸载后必须启动新进程。

### 6.2 完整契约与打包门禁

CPack 在 staging 后、生成 run 包前调用 `_op_abi.py --write-manifest`。工具用 `nm -D` 枚举真实 `aclnn*` 导出，要求每个导出都有可解析的公共契约，且 GetWorkspaceSize 与执行入口配对；随后把契约、host 库与所有公共头的 SHA256 写入 `abi_manifest.json`。无头文件、未覆盖导出、未知类型或不完整的两阶段接口都会使打包失败。此检查对所有 custom run 包执行；Stable-only wheel 不再提供关闭 launcher 的构建模式。

run 包的安装和升级脚本必须复制该清单。wheel staging 对安装后的清单再次校验，避免出现“归档内有清单，但实际安装时丢失”的产物。

类型解析保留基础类型和每一级指针，明确区分 `double` / `double*`、`aclTensor*` / `aclTensor**`，拒绝未支持的声明语法。参数名及 cv 限定不参与 C 调用 ABI 比较。当前只有两类显式兼容项：`SolveTri` 头未公开 C 接口，采用已核对源码的冻结契约；`CausalConv1d` 和 `CausalConv1dBwd` 的 Tensor 变体共用各自的执行入口。修改这些接口必须同步修改契约和测试，不能增加泛化的“缺头跳过”。

`build.sh` 在 package 后继续检查 Stable-ABI adapter 与 public 头是否一致。scoped 构建按 aclnn 符号与算子名关联 adapter，也支持原有文件名关联；`causal_conv1d` 即使实现位于 `stable_causal_conv1d_common.cpp` 也必须检查。指定算子找不到 call site 时失败，不能以零检查成功退出。

### 6.3 安装、运行校验与损坏恢复

安装器在写入 overlay 前校验清单和文件 SHA256，再比较 overlay/base 的完整两阶段接口；import 在加载 vendor 动态库前重复校验。旧 base wheel 可以使用其公共头作为兼容基线；overlay 必须有新清单，旧 overlay 需要重新构建。缺失或损坏的清单、缺头、库或头被替换、base 未声明的符号、参数类型漂移均拒绝启用。

清单用于发现构建遗漏和产物不一致，不是数字签名，也不能证明 kernel 算法或内部 tiling ABI 正确。运行校验不执行 `nm`，只使用 Python 标准库；读取并计算 host 库 SHA256 是启用 overlay 后的额外启动开销。

`fla_npu_opp_manage` 是与 `fla_npu` 并列安装的顶层模块，按自身安装路径加载管理器，不导入 `fla_npu.__init__` 或 vendor 库。即使损坏的 overlay 让 `import fla_npu` 失败，`--status` 仍能报告 `ABI: INVALID`，`--remove-overlay` 仍可回滚。原 `python -m fla_npu.install_opp` 在健康环境保留兼容，但恢复操作应使用新的独立入口。

### 6.4 构建调度、中断与重试

构建调度的进程边界也属于产物正确性的一部分：

- `build.sh -jN` 负责一次构建内部的并行编译；`ci/pkg/make_packages.py` 可并行运行 arm/x86 队列，同一架构内的多个 SoC 仍串行使用独占的 build 目录；
- 本地 Docker 构建和远程 x86 构建都必须放在独立 process group 中；用户中断时向整个进程组转发 `SIGINT`，必要时升级到 `SIGTERM`/`SIGKILL`；
- 远程使用 `setsid` 建立进程组，内部 `timeout --foreground` 保持编译子树在同组；记录 PGID 与进程启动时间，发信号前检查 PID 是否被复用；
- 判断退出时检查整个进程组，忽略僵尸状态，不能只等待组长；本地 Docker 失败或中断时按本次唯一容器名执行 `docker rm -f`，因为容器进程由 Docker daemon 管理；
- 进程组退出后才清理 `build/`、`build_out/`、`third_party/` 和本轮 wheel 输出；没有 `.run`/wheel 的完整产物不得进入成功结果；
- 清理失败使用不可重试的 `BuildCleanupError`，调度器停止后续重试和 SoC 组合，避免仍有写入者时清理目录；
- `build.sh` 单次调用不负责自动重试，CI 调度器默认最多重试三次；每次重试都从清理后的目录重新开始；用户中断会停止后续组合，不把中断当成自动重试。

这条边界避免顶层 shell 已退出但 CMake、make 或 `asc_opc` 子进程仍继续写共享目录，导致下一次重试读到半成品。

### 6.2 CLI

| 参数 | 行为 |
|---|---|
| `--overlay --source <vendor-or-parent>` | 安装一个包内 overlay |
| `--overlay-name <name>` | 指定名称；默认 `fla_npu_overlay_transformer` |
| `--remove-overlay [--overlay-name <name>]` | 删除指定 overlay 并刷新 `config.ini` |
| `--status` | 输出 vendor 顺序、类型、文件数和算子列表 |
| `--install-path <dir> [--force]` | 保留原有外部 OPP 拷贝能力 |

### 6.3 写盘与安全边界

overlay 名称必须匹配：

```text
fla_npu_overlay_[A-Za-z0-9][A-Za-z0-9_.-]*
```

安装器同时执行以下保护：

1. 拒绝绝对路径、路径分隔符、`..`、配置分隔符和 base 名称；
2. 解析后再次确认目标仍位于 `opp/vendors/` 下；
3. 只创建或替换目标 overlay 目录，不修改 base；
4. 删除操作只允许命中合法 overlay；
5. 移除自定义 vendor 中可能遮蔽 CANN 的 `libopapi.so` 别名；
6. overlay 不写入 pip `RECORD`，由 `--remove-overlay` 管理生命周期。

正常安装只新增 `vendors/<overlay_name>/` 并更新 `config.ini`。base 内容和指纹保持不变。

## 7. 构建范围与依赖闭包

本方案的“增量”是**算子范围裁剪**，不是复用旧对象文件：每次仍从干净构建状态生成指定算子的完整 vendor 产物。

`--pkg --jit` 是另一种产物契约：它省略预编译 device kernel `.o`，依赖运行环境完成首次 JIT。当前 wheel staging 校验器要求 AOT 的 kernel `.o` 与 binary-info，因此 JIT run 包不能直接按 AOT wheel 规则验收。A5/CANN 9.2 的实测中，PR 前 `7e39f966` 与当前 PR 的同算子 JIT vendor 文件集合仅差 README，二者首调都返回 `161001`；该结果记录为既有 JIT 路径边界，不作为 overlay 逻辑回归。

| 改动类型 | `--ops` 范围 |
|---|---|
| 算子内部 host/tiling/kernel 实现，公开接口不变 | 只构建该算子 |
| 导出的公共头或 aclnn 接口签名变化 | 同时构建所有依赖方，并更新 schema、Stable wrapper/C++ adapter 和测试 |
| 依赖方内嵌了被改算子的实现拷贝，且希望该拷贝同步变化 | 显式加入依赖方 |
| 仅算子内部使用的 kernel 常量或结构 | 不扩展到其他算子 |

Stable-ABI adapter 不属于 vendor overlay。只要 aclnn 接口不变，修改 host/tiling/kernel 不需要重编 launcher；接口、输出或参数编组发生变化时，必须同步更新并重编 launcher（见 §5.4）。

## 8. 关键不变量

实现和评审必须维持以下不变量：

1. **完整 base 始终存在且不被 overlay 生命周期修改。**
2. **所有 custom 来源都位于 wheel 包内，并通过解析后的路径边界校验。**
3. **L1 与 L2-L4 的 overlay/base 顺序一致。**
4. **Stable launcher 必须显式声明有序多库能力；不允许任何旧 launcher 混用 base host 与 overlay tiling/kernel。**
5. **`ASCEND_CUSTOM_OPP_PATH` 只含 vendor 目录，包内条目由代码整体归一化。**
6. **未设置 `FLA_NPU_OVERLAY_VENDOR` 表示自动发现，显式空值表示禁用 overlay。**
7. **安装和卸载后需要新进程，不能承诺热更新。**

## 9. 兼容性与回滚

| 场景 | 行为 |
|---|---|
| 未安装 overlay | 只使用 base，与原行为一致 |
| 安装 overlay、未设置 `FLA_NPU_OVERLAY_VENDOR` | 按 `config.ini` 与目录自动发现 |
| `FLA_NPU_OVERLAY_VENDOR` 显式为空 | 禁用全部 overlay，只使用 base |
| 多个 overlay | 环境变量已设置时按其中的名称顺序；未设置时按 `config.ini` 顺序逐项覆盖、逐项回落 |
| 旧 launcher | 启用 overlay 后，首次加载缺少有序多库能力的实际 launcher 时硬失败；独立 manager 可审计/卸载并重启恢复 |
| `pip install --force-reinstall` / wheel 升级 | wheel 可能恢复 base 内容；通过 `--status` 审计并重新安装 overlay |
| 回滚单个 overlay | `--remove-overlay --overlay-name <name>`，然后重启进程 |
| 回滚全部 overlay | 逐个移除；`config.ini` 回到仅 base |

本设计不改变算子公开接口和 run 包的全量覆盖入口，因此既有非 overlay 用户不需要迁移。

## 10. 验收结论

历史脚本、日志、digest 和逐轮排障见[过程记录](OPP叠加层增量设计-过程记录.md)。下表保留此前平台验证的边界，不把旧双后端结果当作本轮 Stable-only 数值验收。

本轮新增门禁覆盖正常/新版导出、缺失/旧版/异常导出、实际 override 路径、base-only、stamp 失败先行以及 root import 不加载 Torch。`test_stable_parity.py` 已改为同一 Stable 后端的 baseline/candidate 对比，复用 `verify_stable_only_results.py`，记录签名、精确输入输出、可选返回和状态版本；旧后端报告、空运行或错误不能通过。代表配方不等同于全 37 算子数值验证。

本轮离线集成测试为 185 PASS、4 SKIP、451 subtests PASS；CI 契约为 130 PASS、28 SKIP，详见[新增验证记录](PR878最终检视修复验证.md#stable-only-合并适配验证2026-10-23)。本次新版本 launcher 的原生编译、显式导出实测、A5 overlay 覆盖/回落/卸载后数值复验仍待补，不能宣称已完成硬件验收。

| 维度 | 验收结论 |
|---|---|
| 覆盖与回落 | overlay 算子的 host 与 kernel 改动可观察；未携带算子继续从 base 运行 |
| base 完整性 | A5/A2 安装、卸载和多 overlay 生命周期中 base 指纹保持不变 |
| 多 overlay | 可并存、可排序、可独立卸载；同名算子由最高优先级来源承载 |
| 历史双后端验证 | 旧版本曾验证 ctypes/Stable 输出一致；本轮不再保留 ctypes 路径，需以同一 Stable 后端 baseline/candidate 重新验收 |
| 路径归一化 | 干净、已有反序路径和旧 root/lib 条目都收敛为 `overlay...:base:external` |
| `FLA_NPU_OVERLAY_VENDOR` 显式为空 | `.pth` 与 runtime 均只选择 base；不会出现 base host 搭配 overlay kernel |
| 生成脚本 | source overlay/base 的先后顺序不影响最终优先级 |
| 安全边界 | 非法 overlay 名称、路径逃逸和 base 删除均被拒绝 |
| ABI 安全 | custom/scoped 构建强制检查所选 adapter/header；安装与 import 拒绝公共 aclnn 原型漂移；首次 launcher load 拒绝缺失或旧版有序多库能力 |
| 构建范围 | 单算子包只包含目标算子；A5/A2 已完成原生构建与运行，A3 已完成交叉编译 |
| 回滚 | 卸载后输出和 vendor 顺序回到基线；新进程生效 |
| 构建中断与重试 | 本地/远程进程组终止后清理半成品，再从干净目录重试；单次 `build.sh` 不自动重试 |
| JIT | 只确认 JIT 包布局不含 `.o`；当前 AOT wheel staging 和 A5 首调不通过，不能宣称 JIT wheel/运行支持 |

此前路径修复的回归结果如下；本轮最终检视修复的新增门禁与产物结果见[修复验证记录](PR878最终检视修复验证.md)：

- `tests/test_opp_overlay.py`：12/12 PASS；
- `tests/test_stable_gates.py`：26 PASS、1 个既有 build-stamp 前置条件 SKIP；
- `tests/test_wheel_environment.py`：34/34 PASS；
- scoped ABI 工具对 base3/delta1 实际头文件检查 3 个 call site，0 mismatch；注入 `bool`/`int64_t` 漂移时按预期返回失败；
- A5 3 base + 1 overlay：干净与遗留反序路径均为 4/4 PASS；
- A5 精简 base3 未声明 overlay 独有符号时 import 按预期硬错误；把该符号加入 base 契约后 3 base + 1 overlay 为 4/4 PASS；
- 显式空 overlay：3/3 base PASS，overlay 独有符号不可见；
- 双向 source：均得到 `overlay:base:external`。

## 11. 风险与约束

| 风险/约束 | 处理 |
|---|---|
| CANN 初始化和动态库句柄会缓存来源 | 明确要求安装、卸载后启动新进程 |
| 旧 launcher 不支持多库 | 启用 overlay 后首次 launcher load 硬失败；升级匹配 launcher 或用独立 manager 移除 overlay 后重启 |
| overlay 改变 aclnn 参数契约 | 编译期优先由 scoped gate 拦截，安装期比较 overlay/base 公共头，运行期再次校验；任一不一致均失败 |
| pip 不管理 overlay 文件 | 使用 `--status` 与 `--remove-overlay`；wheel 重装后重新审计 |
| 其他 CANN 版本的多 vendor 行为可能变化 | 新 CANN 版本需要复跑“覆盖 + 回落 + 多 overlay”验收 |
| A3 尚无真机运行验证 | 保留交叉编译证据，真机接入后补运行验收 |
| interface/schema 变化超出普通单算子实现改动 | 按 §7 扩展依赖闭包并同步 Stable wrapper/schema/adapter |

## 12. PR 前行为兼容性自查

初版自查基线为 `7e39f9668eed285fc91cde0502f46cd48686a681`；本表已按主干 `8762b613` 的 Stable-only 收敛提交 `521e1a63` 更新。历史实验结论不自动等同于新合并版本的验收。审计范围覆盖命令行与环境变量、CMake/编译目标、产物名称与目录、安装写盘边界、运行时后端选择、退出条件、性能开销、算子源码和 CI 调用链。

结论使用四种标记：

- **不变**：相同输入得到相同命令路径、产物位置和默认选择；
- **兼容扩展**：旧用法不需修改，新能力只在 overlay 场景出现；
- **有意变化**：为修复路径或接口安全问题而新增的顺序/失败条件；
- **不适用**：该路径不生成 overlay 所需的完整 custom OPP，不执行 overlay 门禁。

### 12.1 构建命令与目标

| PR 前命令/开关 | 本设计后的行为 | 兼容性判定 |
|---|---|---|
| `bash build.sh --pkg --soc=<soc> -jN` | 参数解析、clean、CMake 配置、`package` 目标和 `.run` 生成顺序不变；package 成功后默认增加 stable adapter/public aclnn 头门禁 | **命令兼容，有意新增失败条件**：接口不一致时返回非零 |
| `bash build.sh --pkg --ops=<op>[,<op>...] ...` | 仍按原过滤器只编指定算子；门禁使用同一 `--ops` 集合，只检查对应 adapter/header | **命令兼容**，无需新增参数 |
| `bash build.sh --ops=<op> ...`（不带 `--pkg`） | 仍只执行非打包 custom 构建；`ENABLE_BUILD_PKG=FALSE`，不运行 post-package ABI 门禁 | **不变** |
| `--opkernel`、`--ophost`、`--opapi` | 仍进入各自编译目标，不生成完整 custom run 包，不运行 overlay ABI 门禁 | **不变/不适用** |
| `-u/--test`、各类 `--*_test`、`--run_example` | 仍在测试/example 分支提前退出，不运行 package ABI 门禁 | **不变** |
| `--static` / static package | 保持原 static 构建与打包分支，不经过 `build_package()` 中的 overlay 门禁 | **不变/不适用** |
| `--jit`（不带 `--pkg`） | 不满足完整 custom package 条件，不运行门禁 | **不变/不适用** |
| `--pkg --jit --ops=...` | 保持 host-only/JIT package 构建；若生成 public aclnn 头，则 package 后按 scoped adapter 校验 | **命令兼容**，接口漂移新增失败 |
| 多 SoC `--soc=a,b,...` | 每个 SoC 仍按原循环配置和打包；每份 custom package 完成后执行同一 ABI 门禁 | **产物流程不变**，增加逐包校验 |
| `FLA_NPU_BUILD_STABLE_ABI=0` | wheel 构建立即拒绝；直接 `build.sh --pkg` 也不能以此跳过 adapter/header 门禁 | **有意移除**旧无 launcher 构建路径 |
| 非法 `--ops` 名称 | 仍由既有前置校验在 CMake 前拒绝 | **不变** |
| 直接调用 CMake/Make 目标 | custom CPack package 执行清单门禁；单独编译目标不生成清单，也不执行 `build.sh` 的 adapter 门禁 | 应使用推荐的完整打包入口 |

门禁解析已生成的源码/头文件、读取 ELF 导出并计算产物哈希，不增加 CXX、opc 或 kernel 编译次数；检查分别发生在 CPack staging 后与 package 完成后。

### 12.2 构建产物与包布局

| 产物/契约 | PR 前 | 本设计后 | 判定 |
|---|---|---|---|
| base vendor 物理目录 | `fla_npu_transformer` | 名称和位置不变 | **不变** |
| run 包名称与格式 | `fla_npu_linux-<arch>.run` | 不变；仍由原 CPack/run package 流程生成 | **不变** |
| wheel 名称、tier、platform tag | 既有 A2/A3/A5 分发规则 | 不因 overlay 改名；仍为原 tier/platform 规则 | **不变** |
| wheel launcher | `fla_npu/libfla_npu_stable.so` | 文件名和位置不变；内部 `Runtime` 新增 `FLA_NPU_OP_API_LIBS` 多句柄能力，并保留单数变量兼容 | **兼容扩展** |
| 无 launcher 的 wheel | 历史 ctypes 模式 | 构建、finalize 与发布检查均拒绝；launcher、有效 `_stable_hash.py` 和 Torch/Torch-NPU 下限依赖必须齐全 | **有意移除** |
| wheel Python 文件 | 原运行模块 | `_op_abi.py` 校验导出契约与哈希；新增顶层 `fla_npu_opp_manage.py` 提供独立恢复入口 | **兼容扩展**；不改变公开算子 API |
| vendor ABI 清单 | 不存在 | 新包包含 `abi_manifest.json`；旧 base 可兼容，旧 overlay 需重建 | **新增产物与 overlay 失败条件** |
| base OPP 内容 | 完整聚合 host/索引/tiling/kernel | overlay 安装器不修改 base；完整 wheel/run 包仍走原 base 产物 | **不变** |
| overlay vendor | 不存在 | 新增 `fla_npu_overlay_*` 同级目录，只含增量算子 | **新增功能** |
| torch/aclnn/schema | 既有公开接口 | 本 PR 不修改算子 `def`、aclnn 参数或 torch schema | **不变** |
| 原算子 kernel 源码 | PR 分支原有内容 | 本 PR 净 diff 不包含 `fla/ops/ascendc`；验证期间为对照 `7e39f966` 引入的 4 个 GDN/KDA 文件已移出本 PR | **不变** |

### 12.3 安装、卸载与写盘边界

| 安装路径 | 本设计后的默认行为 | 判定 |
|---|---|---|
| 既有 run 包 `--full`/默认安装 | 主安装器流程与算子支持状态输出不因 overlay CLI 改写 | **不变** |
| run 包 `--install-path=<dir>` | 仍解出 `<dir>/vendors/fla_npu_transformer`；不执行 overlay/base ABI 比对 | **不变** |
| `python -m fla_npu.install_opp --install-path` | 保持外部 OPP 拷贝语义，不进入 overlay 校验 | **不变** |
| `python -m fla_npu_opp_manage --overlay` | 写入独立 overlay 目录和 `config.ini`，写盘前校验清单、哈希和 overlay/base 公共契约 | **新增功能 + 安全条件** |
| `--remove-overlay` | 只删除合法 overlay 并刷新 `config.ini`，拒绝 base 名称和路径逃逸 | **新增功能** |
| pip `RECORD` | 完整 wheel/run 覆盖流程仍按既有 finalizer 刷新；独立 overlay 不写入 pip 账本 | base 流程**不变**；overlay 由专用 CLI 管理 |
| `libopapi.so` 别名 | 旧包可能遗留 custom alias | 继续由既有安装/import 清理逻辑移除，避免遮蔽 CANN `libopapi.so` | **不变** |

### 12.4 环境变量与启动脚本

| 项目 | 无 overlay 时 | 有 overlay 时 | 判定 |
|---|---|---|---|
| `ASCEND_CUSTOM_OPP_PATH` | 当前 wheel 的 base vendor 位于最前，外部 vendor 保持相对顺序 | `overlay... -> base -> external` | vendor-only 和确定性顺序是**有意修复** |
| `<pkg>/opp` 条目 | 不再写入 | 不再写入 | 该 root 对此变量不必要且不充分，删除是**有意修复** |
| `vendor/op_api/lib` 条目 | 不再写入 `ASCEND_CUSTOM_OPP_PATH` | 同左 | host 库由绝对路径加载，不依赖该错误条目 |
| `LD_LIBRARY_PATH` | 本设计不因 OPP overlay 注入 custom op_api/lib | 同左 | 保持包内绝对路径加载边界 |
| `FLA_NPU_OP_API_LIB` | 指向完整 base | 仍指向完整 base；overlay 使用复数列表且强制校验 launcher 能力 | **不变** |
| `FLA_NPU_OP_API_LIBS` | `[base]` | `[overlay..., base]` | **新增环境变量**，当前 launcher 消费 |
| `FLA_NPU_OVERLAY_VENDOR` 未设置 | 无目录时只用 base | 从 `config.ini`/合法目录自动发现 | **新增功能** |
| `FLA_NPU_OVERLAY_VENDOR=""` | 只用 base | 显式禁用全部 overlay，只用 base | **确定语义**，`.pth` 与 runtime 一致 |
| `FLA_NPU_DISABLE_PTH=1` | 继续跳过启动钩子 | 同左 | **不变** |
| `set_env.bash` 重复/source 顺序 | 归一化当前包条目，重复执行幂等 | base/overlay 任意 source 顺序均收敛到配置优先级 | **有意修复**，不影响编译产物 |

注意：路径归一化会把原先错误的 `base -> overlay` 修正为 `overlay -> base`，并把当前 wheel 的 vendor 放在 external 前；这属于功能正确性修复，不应标记为“行为完全不变”。

### 12.5 Python import 与后端选择

| 场景 | 本设计后的行为 | 判定 |
|---|---|---|
| 无 overlay，默认 stable | 仍使用 wheel launcher，L1 为 `base -> CANN`，L2-L4 为 `base -> built-in` | **不变** |
| `FLA_NPU_STABLE_ABI=ctypes` | import/解析前报错，不能选择已删除的后端 | **有意移除** |
| 无 launcher 或有效 build stamp | 构建/发布拒绝；运行期首次 load 硬失败 | **有意新增失败条件** |
| 当前 launcher + overlay | stable 保持默认后端，L1/L2-L4 均按 overlay 到 base | **新增功能** |
| 历史 launcher + overlay | 首次实际 load 检查导出版本，缺失或小于 1 即硬失败 | **有意新增失败条件** |
| `FLA_NPU_STABLE_LIB` 覆盖 launcher | 检查覆盖路径实际加载的库及其 stamp/能力，不能借包内另一份库绕过门禁 | **覆盖路径同等校验** |
| overlay public aclnn 契约漂移 | 安装阶段拒绝；手工绕过安装器时 import 再次拒绝 | **新增失败条件** |
| overlay 不存在或显式禁用 | `_validate_overlay_abi()` 不遍历 overlay，不读取 overlay 头 | base-only 选择**不变** |
| 安装/卸载后同一进程 | 已加载句柄和符号缓存不热切换 | **既有动态库约束**；仍要求新进程 |

### 12.6 失败条件、开销与可观测性

| 维度 | 变化 |
|---|---|
| 新增构建失败 | custom 包缺头、导出未覆盖或两阶段不配对；stable adapter 与头的参数数量、顺序或类型不一致；scoped 算子无 call site |
| 新增安装/import 失败 | overlay 清单缺失、库/头哈希不符、base 未声明符号，或同名符号公共原型不同 |
| 验证边界 | 仅保留已核对的 SolveTri 冻结契约；“类型相同但业务语义变化”和内部 tiling ABI 仍需行为测试 |
| 构建开销 | staging 后读取 ELF 导出、头文件并计算哈希，package 后比较 adapter；不增加 CXX/opc 次数 |
| import 开销 | base-only 不解析 overlay；启用 overlay 时核验库/头 SHA256 和完整公共契约 |
| 诊断 | 路径被归一化时告警一次；Stable 选择可通过 `FLA_NPU_STABLE_TRACE=1` 查看；能力不足列出实际 launcher 路径与独立恢复命令；ABI 错误列出具体符号和参数类型差异 |

### 12.7 源码、CI 与验证结论

- 本 PR 净 diff 不包含 `fla/ops/ascendc`；验证期间为对照 `7e39f966` 临时同步的 4 个 GDN/KDA 文件已移除，不借 overlay 设计修改已有 kernel 实现。
- 导出/清单门禁位于 CPack staging 后，adapter 门禁位于 `build.sh` package 后；CI 复用这些构建入口。
- 此前 `822d31d0` 的契约、安装、环境和进程控制回归为 93 PASS、1 个 build-stamp 前置条件 SKIP；A5 全量 run 安装后 74 个导出全部覆盖，adapter 检查 29 个调用点、0 mismatch；新 wheel 的覆盖、回落和恢复通过代表算子验证。
- `ci/tests/test_run_checks_stage_pipeline.sh` 的 mock 安装路径失败可在未修改的 `a877211a` 同样复现，判定为既有测试夹具问题，不是本轮 ABI/兼容性改动引入。
- 尚未重跑完整 NPU CI 7 分项；A3 仍只有交叉编译证据，没有真机运行结论。

## 13. 实现位置

| 文件 | 职责 |
|---|---|
| `torch_custom/fla_npu/fla_npu/__init__.py` | overlay 解析、包内边界、公共 ABI 校验、路径归一化与 host 库列表；不导入 Torch |
| `torch_custom/fla_npu/fla_npu/_op_abi.py` | 严格解析公共契约、生成清单、校验哈希并比较 overlay/base ABI |
| `cmake/write_opp_abi_manifest.cmake` | CPack staging 后生成并校验 ABI 清单 |
| `torch_custom/fla_npu/fla_npu_opp_manage.py` | 不导入运行时的独立安装/诊断/回滚入口 |
| `torch_custom/fla_npu/fla_npu_opp_env.py` | `.pth` 启动阶段解析 `FLA_NPU_OVERLAY_VENDOR` 并预置路径 |
| `torch_custom/fla_npu/fla_npu/install_opp.py` | overlay 安装、卸载、状态与 `config.ini` 维护 |
| `torch_custom/fla_npu/csrc/src/runtime.cpp` | 多句柄加载、逐符号回落与 `fla_npu_stable_overlay_order_version()` 显式版本导出 |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py` | 实际 launcher 的 stamp 与有序多库能力校验，成功后才解析算子 |
| `tests/test_overlay_launcher_capability.py` | 能力导出版本、失败顺序、override 与 base-only 回归 |
| `torch_custom/fla_npu/csrc/include/stable/runtime.h` | launcher 多句柄状态 |
| `setup.py` | wheel 内 base OPP 的环境脚本生成 |
| `scripts/package/ops_transformer/scripts/finalize_wheel_opp.py` | run 包写入 wheel 后刷新环境脚本与 `RECORD` |
| `tests/test_opp_overlay.py` | overlay 环境变量、自动发现、顺序和安装安全回归 |
| `tests/test_wheel_environment.py` | wheel 环境、生成脚本和运行时加载契约回归 |
| `build.sh` | 全量与 `--ops` scoped 包构建后的 Stable-ABI adapter/header 门禁 |
| `ci/run_checks.sh` | ABI、环境与构建验收接入 |
| `ci/pkg/make_packages.py` | 本地/远程构建 process group、信号转发和清理后重试 |

## 14. 术语

| 术语 | 含义 |
|---|---|
| base vendor | wheel 内完整算子包 `fla_npu_transformer` |
| overlay vendor | 与 base 同级、只包含本次改动算子的高优先级 vendor |
| 逐项回落 | 当前来源没有某个符号/算子时继续查下一个来源，而不是替换整棵树 |
| 聚合产物 | 一份文件承载多个算子，例如 `libcust_opapi.so` 与两个索引 JSON |
| Stable-ABI launcher | `libfla_npu_stable.so`，当前版本支持有序多 custom 句柄 |
| external 条目 | `ASCEND_CUSTOM_OPP_PATH` 中不属于当前 wheel `<pkg>/opp/` 的其他 custom vendor |
