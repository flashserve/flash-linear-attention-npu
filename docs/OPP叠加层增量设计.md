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
| 旧版 Stable-ABI launcher | wheel 内的高性能 C++ aclnn 调用库 `libfla_npu_stable.so`；旧版 `Runtime::symbol()` 只查 `FLA_NPU_OP_API_LIB` 指向的一个 custom 库 | overlay 启用且未显式设置 `FLA_NPU_STABLE_ABI` 时自动改走 ctypes；显式设置为 `stable` 等非 `ctypes` 值时保留 launcher 并明确告警 |
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
site-packages/fla_npu/opp/vendors/
├── config.ini
├── fla_npu_overlay_transformer/       # 本 PR 新增的默认 overlay vendor
│   ├── op_api/lib/libcust_opapi.so
│   ├── op_impl/.../config/<soc>/aic-<soc>-ops-info.json
│   ├── op_impl/.../kernel/<soc>/<op>/*.o
│   ├── op_impl/.../op_tiling/
│   └── op_proto/
└── fla_npu_transformer/               # PR 前已有的完整 base，名称保持不变
```

`fla_npu_transformer` 不是本 PR 新起的名称：既有构建流程使用逻辑 vendor 名 `fla_npu`，打包时追加固定 `_transformer` 后缀形成这个物理目录。本设计不重命名 base，只新增带 `overlay` 标识的 `fla_npu_overlay_transformer`。

`config.ini` 的优先级示例：

```ini
load_priority=fla_npu_overlay_transformer,fla_npu_transformer
```

### 4.1 各层的有序来源

| 层 | 最终顺序 | miss 行为 |
|---|---|---|
| ctypes host 符号 | `overlay_lib... -> base_lib -> libopapi.so` | 继续下一个句柄 |
| Stable-ABI host 符号 | `FLA_NPU_OP_API_LIBS` 中的 `overlay_lib... -> base_lib -> libopapi.so` | 继续下一个句柄 |
| CANN 自定义算子清单/tiling/kernel | `ASCEND_CUSTOM_OPP_PATH` 中的 `overlay_vendor... -> base_vendor`，最后由 CANN built-in 兜底 | 继续下一个 vendor |

同名算子由最高优先级 overlay 覆盖；overlay 未携带的算子在 base 命中。多个 overlay 时，靠前者优先。

### 4.2 关键数据与环境变量

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

### 4.3 `ASCEND_CUSTOM_OPP_PATH` 语义与归一化

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

### 4.4 包内来源边界

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
| 当前 Stable-ABI launcher（默认） | C++ `Runtime::symbol()`：`base -> CANN` | C++ `Runtime::symbol()`：`overlay... -> base -> CANN` |
| 旧版 launcher | 单 base 库可正常使用 | 未显式设置 `FLA_NPU_STABLE_ABI` 时自动切换 ctypes；显式设置为 `stable` 等非 `ctypes` 值时告警，L1 只能从 base 开始查找 |

“启用 overlay”不等同于“目录已经安装”。例如 overlay 目录存在但 `FLA_NPU_OVERLAY_VENDOR=""` 时，overlay 被显式禁用，应走“未启用 overlay”列。

CANN 的 L2-L4 不区分 ctypes/stable：未启用 overlay 时按 `base -> built-in` 查找；启用 overlay 时按 `overlay... -> base -> built-in` 查找。当前两种 L1 后端都使用同一列的来源顺序，从而避免 host 入口与 tiling/kernel 来自不同 vendor。

### 5.4 旧 launcher 兼容

旧 launcher 只识别 `FLA_NPU_OP_API_LIB`，无法按 overlay 到 base 逐项回落。兼容策略为：

- 单数变量始终指向完整 base，避免未覆盖算子出现 `symbol not found`；
- 没有 overlay 时保持既有行为；
- 有 overlay 且 launcher 不支持多库、用户未显式选择后端时，自动选择 ctypes；
- 用户显式要求旧 launcher 时保留选择，但打印明确告警，提示 L1 可能来自 base 而 L2-L4 来自 overlay。

这避免静默形成 host 与 kernel 来源不一致的组合。升级到当前 launcher 后不需要降级。

## 6. 安装、卸载与审计

### 6.1 增量工作流

```bash
# 1. 只构建受影响算子
bash build.sh --pkg --soc=<soc> --ops=<op>[,<dependent>...] -j64

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

Stable-ABI adapter 不属于 vendor overlay。只要 aclnn 接口不变，修改 host/tiling/kernel 不需要重编 launcher；接口、输出或参数编组发生变化时，必须同步更新并重编 launcher。

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
| 旧 launcher | 自动走 ctypes，或在用户显式选择时告警 |
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
| 构建范围 | 单算子包只包含目标算子；A5/A2 已完成原生构建与运行，A3 已完成交叉编译 |
| 回滚 | 卸载后输出和 vendor 顺序回到基线；新进程生效 |

与本次路径修复直接相关的回归结果：

- `tests/test_opp_overlay.py`：10/10 PASS；
- `tests/test_wheel_environment.py`：31/31 PASS；
- A5 3 base + 1 overlay：干净与遗留反序路径均为 4/4 PASS；
- 显式空 overlay：3/3 base PASS，overlay 独有符号不可见；
- 双向 source：均得到 `overlay:base:external`。

## 11. 风险与约束

| 风险/约束 | 处理 |
|---|---|
| CANN 初始化和动态库句柄会缓存来源 | 明确要求安装、卸载后启动新进程 |
| 旧 launcher 不支持多库 | 能力探测后自动使用 ctypes；显式选择时告警 |
| pip 不管理 overlay 文件 | 使用 `--status` 与 `--remove-overlay`；wheel 重装后重新审计 |
| 其他 CANN 版本的多 vendor 行为可能变化 | 新 CANN 版本需要复跑“覆盖 + 回落 + 多 overlay”验收 |
| A3 尚无真机运行验证 | 保留交叉编译证据，真机接入后补运行验收 |
| interface/schema 变化超出普通单算子实现改动 | 按 §7 扩展依赖闭包并同步两种后端适配器 |

## 12. 实现位置

| 文件 | 职责 |
|---|---|
| `torch_custom/fla_npu/fla_npu/__init__.py` | overlay 解析、包内边界、路径归一化、host 库列表、旧 launcher 策略 |
| `torch_custom/fla_npu/fla_npu_opp_env.py` | `.pth` 启动阶段解析 `FLA_NPU_OVERLAY_VENDOR` 并预置路径 |
| `torch_custom/fla_npu/fla_npu/install_opp.py` | overlay 安装、卸载、状态与 `config.ini` 维护 |
| `torch_custom/fla_npu/csrc/src/runtime.cpp` | 当前 launcher 的多句柄加载与逐符号回落 |
| `torch_custom/fla_npu/csrc/include/stable/runtime.h` | launcher 多句柄状态 |
| `setup.py` | wheel 内 base OPP 的环境脚本生成 |
| `scripts/package/ops_transformer/scripts/finalize_wheel_opp.py` | run 包写入 wheel 后刷新环境脚本与 `RECORD` |
| `tests/test_opp_overlay.py` | overlay 环境变量、自动发现、顺序和安装安全回归 |
| `tests/test_wheel_environment.py` | wheel 环境、生成脚本和运行时加载契约回归 |
| `ci/run_checks.sh` | ABI、环境与构建验收接入 |

## 13. 术语

| 术语 | 含义 |
|---|---|
| base vendor | wheel 内完整算子包 `fla_npu_transformer` |
| overlay vendor | 与 base 同级、只包含本次改动算子的高优先级 vendor |
| 逐项回落 | 当前来源没有某个符号/算子时继续查下一个来源，而不是替换整棵树 |
| 聚合产物 | 一份文件承载多个算子，例如 `libcust_opapi.so` 与两个索引 JSON |
| Stable-ABI launcher | `libfla_npu_stable.so`，当前版本支持有序多 custom 句柄 |
| external 条目 | `ASCEND_CUSTOM_OPP_PATH` 中不属于当前 wheel `<pkg>/opp/` 的其他 custom vendor |
