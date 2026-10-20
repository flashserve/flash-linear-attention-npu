# FLA NPU 算子包「单算子叠加层」增量设计

- **作者**：zzy1205
- **状态**：已实现；A5（Ascend950PR）与 A2（Ascend910B3）已完成行为验证
- **适用范围**：`flash-linear-attention-npu` wheel 内嵌 OPP 的单算子或少量算子迭代
- **过程归档**：[《OPP 叠加层设计过程记录》](OPP叠加层增量设计-过程记录.md)

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

![运行期四条查找链与分层来源](images/pr878-01-lookup-layers.png)

一次自定义算子调用涉及四条查找链：

| 层 | 查找对象 | 产物 | 选择方 |
|---|---|---|---|
| L1 | host aclnn 符号 | `op_api/lib/libcust_opapi.so` | ctypes 或 Stable-ABI launcher（wheel 内的高性能 C++ aclnn 调用库 `libfla_npu_stable.so`） |
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
- ctypes 与当前 Stable-ABI launcher 使用相同的 host 库顺序。
- CANN 侧用于确认“自定义算子存在”的算子清单，以及后续 tiling、kernel 的查找顺序，与 L1 host aclnn 库保持同一 `overlay -> base` 优先级。
- 支持多个 overlay，顺序明确、可审计、可单独卸载。
- base vendor 在 overlay 安装和卸载过程中保持不变。
- 为旧版 launcher、overlay 启用规则和已有错误路径定义可验证的兼容行为。

| 兼容场景 | 业务含义与代码出处 | 确定行为 |
|---|---|---|
| 旧版 Stable-ABI launcher | wheel 内的高性能 C++ aclnn 调用库 `libfla_npu_stable.so`；旧版 `Runtime::symbol()` 只查 `FLA_NPU_OP_API_LIB` 指向的一个 custom 库 | 未指定后端时改走 ctypes；显式 stable 时先完成 ABI 校验，失败则报错，通过后告警“L1 只能命中 base”并继续 |
| `FLA_NPU_OVERLAY_VENDOR` | `fla_npu/__init__.py` 与 `.pth` 启动钩子读取的环境变量，用于选择启用哪些 overlay 及其优先级 | 未设置时按 `config.ini`/目录自动发现；显式名称列表按给定顺序；显式空值时禁用全部 overlay，只使用 base |
| 已有错误 OPP 路径 | shell 可能已把 base 放在 overlay 前面，或遗留 `<pkg>/opp`、`vendor/op_api/lib` 条目 | 自动清理当前包的旧条目并归一化为 `overlay -> base -> external`；实际发生修正时告警一次 |

### 3.2 非目标

- 不提供进程内热更新；安装或卸载后必须启动新进程。
- 不把 CMake 对象缓存改造成通用对象级增量构建系统。
- 不放宽 host 库只能从当前 wheel 包内加载的边界。
- 不修改算子的 `def`、aclnn 或 torch schema。
- 不改变既有 run 包的 `--full` 覆盖安装语义。

## 4. 总体架构

![叠加层架构与数据流](images/opp-overlay-arch.png)

```text
site-packages/fla_npu/
├── libfla_npu_stable.so               # wheel 级 launcher，不属于任何 vendor
└── opp/vendors/
    ├── config.ini
    ├── fla_npu_overlay_transformer/   # 本 PR 新增的默认 overlay vendor
    │   ├── op_api/lib/libcust_opapi.so
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

### 4.1 行为相关产物清单

| 产物 | 所在位置 | 作用与归属 |
|---|---|---|
| `.pth` 启动钩子 | `site-packages/fla_npu_opp_env.pth` + `.py` | 在 CANN 初始化前选择并排列 vendor |
| Python 分发与 wrapper | `fla_npu/__init__.py`、`fla_npu/ops/ascendc/*.py` | 解析来源、选择 ctypes/launcher、组织调用参数 |
| launcher build-stamp | `fla_npu/ops/ascendc/_stable_hash.py` | 确认 launcher 二进制与同 wheel adapter 源码匹配 |
| Stable-ABI launcher | `fla_npu/libfla_npu_stable.so` | wheel 级 L1 C++ 调用入口，所有 vendor 共用 |
| vendor 优先级 | `opp/vendors/config.ini` | 记录 overlay 到 base 的默认顺序 |
| shell 环境脚本 | `<vendor>/bin/set_env.bash` | source 时归一化 `ASCEND_CUSTOM_OPP_PATH` |
| public aclnn 头 | `<vendor>/op_api/include/aclnnop/*.h` | 编译、安装和运行期可检查的 L1 ABI 契约 |
| custom host 库 | `<vendor>/op_api/lib/libcust_opapi.so` | L1 aclnn 实现 |
| 算子发现清单 | `<vendor>/op_impl/.../config/<soc>/aic-*-ops-info.json` | L2：CANN 确认自定义算子存在 |
| 实现源码/JIT 支撑 | `<vendor>/op_impl/.../<vendor>_impl/{dynamic,ascendc}/**` | 随包携带的实现源码、公共头和动态/JIT 支撑文件；不单独决定 overlay 顺序 |
| kernel 索引 | `<vendor>/op_impl/.../kernel/config/<soc>/binary_info_config.json` | L3：选择 kernel 变体 |
| tiling 库 | `<vendor>/op_impl/.../op_tiling/lib/liboptiling.so` 等 | host 侧生成 tiling data |
| device kernel 与旁车元数据 | `<vendor>/op_impl/.../kernel/<soc>/<op>/{*.o,*.json}` | L4：NPU 执行实体及其二进制描述 |
| proto/ES | `<vendor>/op_proto/**`、`libcust_opsproto_rt2.0.so`、`libes*.so` | 算子原型、proto 与表达式相关产物 |
| CANN 兜底 | CANN 安装目录中的 `libopapi.so` 与 `opp/built-in` | wheel 外部的最后一级 L1、L2-L4 回落来源 |

架构图独立展示上述会参与来源选择、ABI 检查或实际执行的产物；README、`version.info`、pip 元数据等不改变查找决策的文件不展开。

### 4.2 各层的有序来源

| 层 | 最终顺序 | miss 行为 |
|---|---|---|
| ctypes host 符号 | `overlay_lib... -> base_lib -> libopapi.so` | 继续下一个句柄 |
| Stable-ABI host 符号 | `FLA_NPU_OP_API_LIBS` 中的 `overlay_lib... -> base_lib -> libopapi.so` | 继续下一个句柄 |
| CANN 自定义算子清单/tiling/kernel | `ASCEND_CUSTOM_OPP_PATH` 中的 `overlay_vendor... -> base_vendor`，最后由 CANN built-in 兜底 | 继续下一个 vendor |

同名算子由最高优先级 overlay 覆盖；overlay 未携带的算子在 base 命中。多个 overlay 时，靠前者优先。

### 4.3 关键数据与环境变量

| 名称 | 语义 |
|---|---|
| `vendor_dirs` | 当前 wheel 内的有序 vendor 列表：`[overlay..., base]` |
| `FLA_NPU_OVERLAY_VENDOR` | 可选的 `os.pathsep` 分隔 overlay 名称列表；设置后覆盖自动发现顺序 |
| `FLA_NPU_OP_API_LIBS` | 当前 launcher 使用的有序 host 库列表：`overlay... -> base` |
| `FLA_NPU_OP_API_LIB` | 旧 launcher 的兼容输入，始终指向完整 base 库 |
| `ASCEND_CUSTOM_OPP_PATH` | CANN 自定义 OPP vendor 目录列表，只登记 vendor 目录 |
| `FLA_NPU_OPP_PATH` | 当前 wheel 的 OPP root，供诊断使用，不参与 `ASCEND_CUSTOM_OPP_PATH` 查找 |
| `FLA_NPU_STABLE_ABI` | 后端选择；`ctypes` 强制参考路径，其他值选择 launcher |
| `FLA_NPU_DISABLE_PTH` | 禁用解释器启动阶段的 `.pth` 环境预置 |

`FLA_NPU_OVERLAY_VENDOR` 的状态必须严格区分：

| 状态 | 行为 |
|---|---|
| 未设置 | 从 `config.ini` 和包内合法的 `fla_npu_overlay_*` 目录自动发现 |
| 显式名称列表 | 仅使用列表中存在且合法的 overlay，保持用户给定顺序 |
| 显式空字符串、纯空白或纯分隔符 | 禁用所有 overlay，只使用 base |

`.pth` 启动钩子、`import fla_npu` 和安装工具必须遵守同一套 `FLA_NPU_OVERLAY_VENDOR` 选择规则。

### 4.4 `ASCEND_CUSTOM_OPP_PATH` 语义与归一化

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

### 4.5 包内来源边界

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
  -> 应用旧 launcher 兼容策略
  -> 加载 libopapi.so 与各 custom libcust_opapi.so
  -> 缓存 [overlay_lib..., base_lib, cann_lib]
```

ctypes 侧按缓存列表逐符号查找。当前 launcher 读取复数变量 `FLA_NPU_OP_API_LIBS`，逐个 `dlopen` 并按相同顺序 `dlsym`；custom 全部 miss 后才落 CANN。

### 5.3 overlay 状态与 L1 后端组合

overlay 是否启用与 L1 使用哪个调用后端是两个独立维度：

- overlay 状态决定可查找的来源集合；
- ctypes 或 Stable-ABI launcher 决定由 Python 还是 C++ 执行 L1 host 符号查找。

| L1 调用后端 | 未启用 overlay | 启用 overlay |
|---|---|---|
| ctypes | Python `_runtime.symbol()`：`base -> CANN` | Python `_runtime.symbol()`：`overlay... -> base -> CANN` |
| 当前 wheel launcher（支持有序多库，默认） | C++ `Runtime::symbol()`：`base -> CANN` | C++ `Runtime::symbol()`：`overlay... -> base -> CANN` |
| 历史 wheel launcher（只支持单个 custom 库） | 单 base 库可正常使用 | 未指定后端时自动切换 ctypes；显式 stable 时 ABI 失败则报错，ABI 通过后告警 L1 从 base 查找并继续 |

“启用 overlay”不等同于“目录已经安装”。例如 overlay 目录存在但 `FLA_NPU_OVERLAY_VENDOR=""` 时，overlay 被显式禁用，应走“未启用 overlay”列。

CANN 的 L2-L4 不区分 ctypes/stable：未启用 overlay 时按 `base -> built-in` 查找；启用 overlay 时按 `overlay... -> base -> built-in` 查找。当前两种 L1 后端都使用同一列的来源顺序，从而避免 host 入口与 tiling/kernel 来自不同 vendor。

这里的“当前/历史 launcher”描述的是**安装中的 wheel 所携带的 `libfla_npu_stable.so` 能力版本**，不是“base 算子包/overlay 算子包”各自带了一份 launcher。

### 5.4 launcher 复用与检查边界

普通 overlay 更新已有算子的 host/tiling/kernel 实现，但不改变 torch schema、aclnn 参数签名、输出和 mutation 契约时，直接复用 wheel 中的 launcher 是正确行为：launcher 内的 adapter 负责参数编组，再按符号名从 `overlay_lib -> base_lib -> CANN` 查找实际 aclnn 实现。

overlay 没有携带 launcher 不会产生告警，也不存在“overlay 缺 launcher 后再从 base vendor 取一份”的动作；进程从始至终只有 wheel 根目录下那一份 launcher。

| 检查 | 代码位置 | 能保证什么 | 不能保证什么 |
|---|---|---|---|
| 多库能力探测 | `fla_npu/__init__.py::_launcher_supports_overlay_order()` | 当前 launcher 是否识别 `FLA_NPU_OP_API_LIBS`，从而能按 overlay 到 base 查 host 符号；不支持时决定自动 ctypes 或显式 stable 告警 | 不校验某个 overlay 是否改变了 aclnn 参数契约 |
| launcher build stamp | `ops/ascendc/_stable.py::_check_build_stamp()` | wheel 内 launcher 二进制与同一 wheel 的 Python/stable adapter 源码是否匹配 | 不比较 overlay 与 base 的接口差异 |
| 单算子后端选择 | `ops/ascendc/__init__.py::_get_stable_op()` | launcher 未注册该算子时，如存在 ctypes wrapper 就回落 ctypes；可用 `FLA_NPU_STABLE_TRACE=1` 观察 | overlay 不带 launcher 本身不会触发告警 |
| 编译期 aclnn ABI 门禁 | `build.sh`、`tools/op_abi_validate.py` | custom 全量及 `--ops` 单算子构建完成后默认检查 stable adapter 与本次产物 aclnn 头文件的参数数量、顺序和类型；不一致时构建失败 | 不要求用户增加参数；无法识别“类型不变但业务语义改变” |
| 安装期/运行期 vendor ABI 比对 | `install_opp --overlay`、`fla_npu/_op_abi.py`、`fla_npu/__init__.py` | 默认比较 overlay/base 公共 aclnn 原型；新增符号或参数契约不一致时拒绝安装，手工绕过安装器也会在 import 时硬错误 | 没有公开 aclnn 头的已知缺口仍需编译门禁或人工确认 |

因此：

- **已有算子的实现改动，接口不变**：复用 wheel launcher，不需要重编；
- **aclnn/schema/adapter 契约变化**：构建、安装或运行检查会失败；必须更新并重编 wheel launcher/wrapper，通过 ABI 门禁后发布完整 wheel，不能只发布 overlay；
- **全新算子且 base wheel 从未声明**：仅复制一个 OPP overlay 不足以增加 Python 公共入口，需要同步更新 wheel 的算子清单、wrapper 和 launcher/ctypes 承载。

### 5.5 历史 launcher 兼容

旧 launcher 只识别 `FLA_NPU_OP_API_LIB`，无法按 overlay 到 base 逐项回落。兼容策略为：

- 单数变量始终指向完整 base，避免未覆盖算子出现 `symbol not found`；
- 没有 overlay 时保持既有行为；
- 有 overlay 且 launcher 不支持多库、用户未显式选择后端时，自动选择 ctypes；
- 用户显式要求旧 launcher 时，运行时已先完成 overlay/base ABI 比对：不一致直接报错；一致时打印明确告警，说明 L1 只能命中 base，而 L2-L4 仍按 overlay 优先。

ABI 校验阻止不兼容契约进入这个混合来源组合；接口一致时保留旧 launcher 的兼容运行能力。升级到当前 launcher 后不需要降级。

## 6. 安装、卸载与审计

### 6.1 增量工作流

```bash
# 1. 只构建受影响算子
bash build.sh --pkg --soc=<soc> --ops=<op>[,<dependent>...] \
  -j64

# 2. 将 run 包解到隔离目录
build_out/fla_npu_*.run --install-path=<dir> --quiet

# 3. 安装为当前 wheel 内的 overlay
python -m fla_npu.install_opp \
  --overlay \
  --source <dir>/vendors \
  --force

# 4. 审计或回滚
python -m fla_npu.install_opp --status
python -m fla_npu.install_opp --remove-overlay
```

安装或卸载后必须启动新进程。

ABI 校验是 overlay 流程的默认安全步骤，不增加命令参数：custom 包在原构建命令结束前运行 `op_abi_validate.py`，overlay 安装和 import 分别进行 vendor 契约比对。

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

| 改动类型 | `--ops` 范围 |
|---|---|
| 算子内部 host/tiling/kernel 实现，公开接口不变 | 只构建该算子 |
| 导出的公共头或 aclnn 接口签名变化 | 同时构建所有依赖方，并更新 schema、ctypes/launcher 适配器和测试 |
| 依赖方内嵌了被改算子的实现拷贝，且希望该拷贝同步变化 | 显式加入依赖方 |
| 仅算子内部使用的 kernel 常量或结构 | 不扩展到其他算子 |

Stable-ABI adapter 不属于 vendor overlay。只要 aclnn 接口不变，修改 host/tiling/kernel 不需要重编 launcher；接口、输出或参数编组发生变化时，必须同步更新并重编 launcher（见 §5.4）。

## 8. 关键不变量

实现和评审必须维持以下不变量：

1. **完整 base 始终存在且不被 overlay 生命周期修改。**
2. **所有 custom 来源都位于 wheel 包内，并通过解析后的路径边界校验。**
3. **L1 与 L2-L4 的 overlay/base 顺序一致。**
4. **当前 launcher 与 ctypes 都逐符号回落；旧 launcher 不允许静默混用来源。**
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
| 旧 launcher | 未显式选择时自动走 ctypes；显式 stable 时先校验 ABI，失败报错，通过后告警 L1 从 base 查找并继续 |
| `pip install --force-reinstall` / wheel 升级 | wheel 可能恢复 base 内容；通过 `--status` 审计并重新安装 overlay |
| 回滚单个 overlay | `--remove-overlay --overlay-name <name>`，然后重启进程 |
| 回滚全部 overlay | 逐个移除；`config.ini` 回到仅 base |

本设计不改变算子公开接口和 run 包的全量覆盖入口，因此既有非 overlay 用户不需要迁移。

## 10. 验收结论

详细脚本、日志、digest 和逐轮排障见[过程记录](OPP叠加层增量设计-过程记录.md)。规范性结论如下：

| 维度 | 验收结论 |
|---|---|
| 覆盖与回落 | overlay 算子的 host 与 kernel 改动可观察；未携带算子继续从 base 运行 |
| base 完整性 | A5/A2 安装、卸载和多 overlay 生命周期中 base 指纹保持不变 |
| 多 overlay | 可并存、可排序、可独立卸载；同名算子由最高优先级来源承载 |
| 两种后端 | ctypes 与当前 launcher 均按 overlay 到 base 逐符号解析；端到端输出一致 |
| 路径归一化 | 干净、已有反序路径和旧 root/lib 条目都收敛为 `overlay...:base:external` |
| `FLA_NPU_OVERLAY_VENDOR` 显式为空 | `.pth` 与 runtime 均只选择 base；不会出现 base host 搭配 overlay kernel |
| 生成脚本 | source overlay/base 的先后顺序不影响最终优先级 |
| 安全边界 | 非法 overlay 名称、路径逃逸和 base 删除均被拒绝 |
| ABI 安全 | custom/scoped 构建默认检查所选 adapter/header；安装与 import 默认拒绝 overlay/base 公共 aclnn 原型漂移；强制历史 launcher 仅在 ABI 通过后告警继续 |
| 构建范围 | 单算子包只包含目标算子；A5/A2 已完成原生构建与运行，A3 已完成交叉编译 |
| 回滚 | 卸载后输出和 vendor 顺序回到基线；新进程生效 |

与本次路径修复直接相关的回归结果：

- `tests/test_opp_overlay.py`：13/13 PASS；
- `tests/test_stable_gates.py`：26 PASS、1 个既有 build-stamp 前置条件 SKIP；
- `tests/test_wheel_environment.py`：36/36 PASS；
- scoped ABI 工具对 base3/delta1 实际头文件检查 3 个 call site，0 mismatch；注入 `bool`/`int64_t` 漂移时按预期返回失败；
- A5 3 base + 1 overlay：干净与遗留反序路径均为 4/4 PASS；
- A5 精简 base3 未声明 overlay 独有符号时 import 按预期硬错误；把该符号加入 base 契约后 3 base + 1 overlay 为 4/4 PASS；
- 显式空 overlay：3/3 base PASS，overlay 独有符号不可见；
- 双向 source：均得到 `overlay:base:external`。

## 11. 风险与约束

| 风险/约束 | 处理 |
|---|---|
| CANN 初始化和动态库句柄会缓存来源 | 明确要求安装、卸载后启动新进程 |
| 旧 launcher 不支持多库 | 未显式选择时自动 ctypes；显式 stable 时 ABI 失败报错，通过后告警混合来源并继续 |
| overlay 改变 aclnn 参数契约 | 编译期优先由 scoped gate 拦截，安装期比较 overlay/base 公共头，运行期再次校验；任一不一致均失败 |
| pip 不管理 overlay 文件 | 使用 `--status` 与 `--remove-overlay`；wheel 重装后重新审计 |
| 其他 CANN 版本的多 vendor 行为可能变化 | 新 CANN 版本需要复跑“覆盖 + 回落 + 多 overlay”验收 |
| A3 尚无真机运行验证 | 保留交叉编译证据，真机接入后补运行验收 |
| interface/schema 变化超出普通单算子实现改动 | 按 §7 扩展依赖闭包并同步两种后端适配器 |

## 12. PR 前行为兼容性自查

自查基线为上游提交 `7e39f9668eed285fc91cde0502f46cd48686a681`，按既有命令、默认值、产物路径、退出条件和算子源码逐项比较。

| PR 前用法/产物 | 本设计后的默认行为 | 结论 |
|---|---|---|
| `bash build.sh --pkg ...` | 命令、CMake/package 目标和产物路径不变；包生成后默认增加 ABI gate | 命令兼容；接口不一致新增明确失败 |
| `bash build.sh --pkg --ops=<op> ...` | 算子过滤、清理、编译和打包路径不变；默认只检查所选算子的 adapter/header | 命令兼容；无需额外参数 |
| NPU CI 全量/scoped 构建 | 复用 `build.sh` 内同一门禁，不再在 CI 外层重复执行 | 无重复检查 |
| `FLA_NPU_BUILD_STABLE_ABI=0` | 仍构建 pure-ctypes wheel，不要求 launcher | 不变 |
| 默认 wheel build | launcher 文件名和位置仍为 `fla_npu/libfla_npu_stable.so`；当前二进制新增多库能力，无 overlay 时仍查 `base -> CANN` | 命令/路径不变，能力向后兼容 |
| base run 包/vendor | 仍产出 `fla_npu_transformer` 与原 `.run` 命名，不重命名 base | 不变 |
| 既有 `install_opp --install-path` | 不触发 overlay ABI 校验；外部 OPP 拷贝行为不变 | 不变 |
| 新增 `install_opp --overlay` | 命令不增加参数；安装前默认比较 overlay/base public aclnn 契约 | 新功能内部安全校验 |
| 无 overlay 的 `import fla_npu` | 仍只加载 base；不执行 overlay ABI 比对 | 算子选择不变 |
| 生成的 `set_env.bash` | 改为只登记 vendor 并归一化顺序，不再登记无效 OPP root/op_api-lib 条目 | 有意的运行环境修复，不改变编译过程 |
| 原算子 kernel 源码 | PR 工作树中的 GDN/KDA 4 个反向差异已对齐 `7e39f966`，不再进入 overlay PR 净 diff | 不改变 PR 前算子实现/编译产物 |

不新增构建或安装参数。既有非 custom 编译模式不运行门禁；custom/overlay 流程默认校验 ABI，是本设计为有序多来源新增的安全条件。

## 13. 实现位置

| 文件 | 职责 |
|---|---|
| `torch_custom/fla_npu/fla_npu/__init__.py` | overlay 解析、包内边界、路径归一化、host 库列表、旧 launcher 策略 |
| `torch_custom/fla_npu/fla_npu/_op_abi.py` | 解析 public aclnn 原型并比较 overlay/base ABI |
| `torch_custom/fla_npu/fla_npu_opp_env.py` | `.pth` 启动阶段解析 `FLA_NPU_OVERLAY_VENDOR` 并预置路径 |
| `torch_custom/fla_npu/fla_npu/install_opp.py` | overlay 安装、卸载、状态与 `config.ini` 维护 |
| `torch_custom/fla_npu/csrc/src/runtime.cpp` | 当前 launcher 的多句柄加载与逐符号回落 |
| `torch_custom/fla_npu/csrc/include/stable/runtime.h` | launcher 多句柄状态 |
| `setup.py` | wheel 内 base OPP 的环境脚本生成 |
| `scripts/package/ops_transformer/scripts/finalize_wheel_opp.py` | run 包写入 wheel 后刷新环境脚本与 `RECORD` |
| `tests/test_opp_overlay.py` | overlay 环境变量、自动发现、顺序和安装安全回归 |
| `tests/test_wheel_environment.py` | wheel 环境、生成脚本和运行时加载契约回归 |
| `build.sh` | 全量与 `--ops` scoped 包构建后的 Stable-ABI adapter/header 门禁 |
| `ci/run_checks.sh` | ABI、环境与构建验收接入 |

## 14. 术语

| 术语 | 含义 |
|---|---|
| base vendor | wheel 内完整算子包 `fla_npu_transformer` |
| overlay vendor | 与 base 同级、只包含本次改动算子的高优先级 vendor |
| 逐项回落 | 当前来源没有某个符号/算子时继续查下一个来源，而不是替换整棵树 |
| 聚合产物 | 一份文件承载多个算子，例如 `libcust_opapi.so` 与两个索引 JSON |
| Stable-ABI launcher | `libfla_npu_stable.so`，当前版本支持有序多 custom 句柄 |
| external 条目 | `ASCEND_CUSTOM_OPP_PATH` 中不属于当前 wheel `<pkg>/opp/` 的其他 custom vendor |
