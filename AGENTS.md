# AGENTS.md

本文件是给 AI coding agent 的仓库级工作说明，适用于整个 `flash-linear-attention-npu` 仓库。若子目录后续出现更近的 `AGENTS.md`，以更近文件为准。

## 项目定位

`flash-linear-attention-npu` 是面向昇腾 NPU 的高性能线性注意力算子库，核心工作包括 Ascend C 算子、Tiling/InferShape/op_host、aclnn op_api、PyTorch 适配、Triton 适配、单算子测试和端到端 Example/ST。

优先阅读：

- `README.md`：构建、安装、调用、测试入口和目录结构。
- `CONTRIBUTING.md`：贡献流程和新增算子交付要求。
- `docs/repository-rules.md`：分支、ABI、NPU CI 和合入规则。
- `docs/agents/foundation.md`：项目分层、调用链、关键术语、shape/layout、chunk 依赖和内存生命周期基础。
- `docs/agents/development/coding-standard.md`：Ascend C 算子的文件分层、命名、kernel/tiling 职责、资源预算和同步生命周期编码规范。
- `docs/agents/optimization/README.md`：算子性能设计与优化的依赖模型、技术分类、SOC 差异和检查清单入口。
- `docs/agents/architecture/torch-npu-decoupled.md`：默认 `fla_npu.ops.ascendc` 解耦运行时、依赖确定与兼容性门禁、wheel 产物、多卡 device guard、stream、数据依赖、autograd 和 ACL 私有格式透传设计。
- `docs/agents/README.md`：面向 AI agent 的开发原理、方法论、验证和经验总结索引。
- `.github/pull_request_template.md`：PR 必填信息和验证矩阵。
- 当前修改算子的 `README.md`、`docs/aclnn*.md`、测试脚本和相邻算子实现。

## 工作原则

- 开始前先看 `git status --short`，不要回滚或覆盖用户已有改动。
- 先用 `rg` / `rg --files` 找代码和文档，再修改；不要凭记忆猜目录。
- 改动保持聚焦，避免无关格式化、批量重排和生成物噪声。
- 公共接口、shape/dtype/layout/range、预留参数、平台差异、返回码或报错文本变化，必须同步检查代码、README、aclnn 文档、PyTorch API 文档、测试和示例。
- 编写或重构 Ascend C 算子代码时必须遵守 `docs/agents/development/coding-standard.md`；涉及 A5 时同时遵守 `docs/agents/optimization/soc/a5.md` 的 A5 编码约束。
- 性能设计或优化前必须先判断 chunk 间是否存在 carry，并按 `docs/agents/optimization/README.md` 阅读对应依赖模型、技术分类和目标 SOC 文档。不得把具体算子的窗口数、slot 数、容量或同步协议未经重新推导直接复制到其他算子。
- 公开 PR、issue、评论和总结中不要暴露内网地址、机器名、用户名、绝对路径、临时目录、日志路径、token 或本地调测环境细节。
- 构建和测试默认面向 Linux + CANN + NPU 环境；其他平台只做静态阅读、文本编辑或格式检查，不把未验证命令写成已验证结论。

## 算子原型和公开接口兼容性红线

本规则统一适用于仓库内所有算子。算子原型、aclnn 接口和 `fla_npu.ops.ascendc.<op_name>` 接口一旦确定，除非仓库管理账号 `@weinachuan` 针对本次具体变更明确下达命令，否则任何功能开发、问题修复、精度或性能优化、tiling、平台适配和重构都不得修改以下内容：

1. `op_host/*_def.cpp` 中已有的输入、输出和属性，包括名称、数量、顺序、必选/可选、默认值和既有语义。允许在不删除或改变现有 dtype 支持和语义的前提下增加 dtype；新增 dtype 时仍需同步代码校验、文档和测试。
2. 算子 aclnn 接口，包括接口参数名称、数量、数据类型、顺序、默认值语义和原始默认行为，必须保持 ABI 和源码调用兼容。
3. `fla_npu.ops.ascendc.<op_name>` 公开接口，包括参数名称、数量、顺序、默认值语义和原始默认行为，必须保证现有用户代码兼容。

任何改动上述原型或接口的想法、方案和试验，都必须在实施、提交代码或修改文档前向 `@weinachuan` 说明并等待明确命令。说明至少包括：拟修改前后的完整接口、为什么一定要修改、为什么不能通过已有输入或属性、InferShape、origin/view/storage shape、tiling data、workspace、内部结构或内部算子让实现获得所需信息，以及 ABI、源码和用户调用兼容性影响。未获得明确命令时不得先改后问，也不得以“内部属性”“带默认值”“只影响 L0”或“调用方可以适配”为由变更。

对于已经发布的原型或接口，默认优先保留原接口并新增独立 V2 实现，由旧接口继续维持原始默认行为；是否新增 V2 以及 V2 的具体原型同样需要 `@weinachuan` 明确确认，不得直接在原接口上做不兼容演进。

## L0 单路径、泛化和实现信息归属红线

本规则适用于 Ascend C 算子的内部 L0 设计、融合和性能优化。L0 即使没有直接暴露给用户，也属于需要长期维护的算子契约和调用路径。

1. **模板覆盖范围**：分别声明性能目标、功能支持范围和模板优势域。需求没有明确限制的维度必须保持泛化；模板边界必须基于核心计算结构或资源规划的实质差异，不能按单个规则 shape 或普通边界形态随意拆分。同一模板应覆盖其完整优势域，并同时保证目标性能和范围内的正确性。
2. **L0 路径统一**：同一算子在所有支持的 SOC 上复用同一个 L0 算子定义、原型和 L2 调用路径。平台和 shape 差异只能在该 L0 内部通过 host tiling、tiling data、workspace 规划、kernel 模板参数或架构 trait 表达；内部可以有多个模板实例，但不能形成多个 L0 契约。
3. **融合路径收敛**：融合实现覆盖既有功能范围、通过精度验证并达到目标性能后，必须删除被替代的未融合 L0 调用路径、注册和构建入口。仍需复用的算法只能迁入统一 L0 的内部模板或组件，不能以第二套 L0 fallback 长期存在。
4. **实现信息内聚**：L0 输入、输出和属性只承载不可由现有语义信息推导的数据。shape、平台能力、模板选择、tile、任务划分和 workspace 等实现信息应由 host tiling 推导，并通过 tiling data、workspace 或编译期模板参数传给 kernel，不得编码成冗余 L0 入参。
5. **设计前置确认**：新增或修改 L0 原型、拆分或融合 L0、增加 V2 L0，或确需保留多条 L0 路径时，必须在实现、文档和测试变更前向 `@weinachuan` 提交设计并取得明确确认。设计说明至少包含修改前后的调用图和原型、复用现有 L0 不可行的原因、信息推导方案、泛化与性能影响、ABI 影响，以及旧路径迁移或删除计划；默认方案是只保留一套 L0 路径。

### KDA 私有 L0 优化授权

针对当前 `ChunkKdaFwd` 性能优化任务，`@weinachuan` 已明确授权调整私有 L0 原型、kernel 阶段边界、融合关系、workspace、L1/UB/L0A/L0B/L0C 分配、MMAD 次数、流水和调度策略。不得用静态测试固化某一种私有实现结构。唯一必须保持的是 `aclnnChunkKdaFwd` 的 L2 ABI、已支持的功能范围与语义，以及 `fla_npu.ops.ascendc.chunk_kda_fwd` 的接口与行为。具有独立公开入口的算子可以优化内部实现，但不得连带修改其公开接口；如确需改变任一公开接口，仍须另行取得明确确认。

## 关键目录

- `fla/ops/ascendc/`：Ascend C 算子实现。
- `fla/ops/ascendc/common/`：公共 Ascend C 组件。
- `fla/ops/ascendc/gdn/`：GDN 相关算子。
- `torch_custom/fla_npu/`：PyTorch 自定义算子适配、YAML schema、Python 包和测试。
- `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py`：推荐 Python 稳定入口导出。
- `examples/`：端到端调用示例。
- `ci/`：NPU CI、Example/ST case 和本地 CI 脚本。
- `scripts/`：构建、打包、环境检查和代码生成辅助脚本。
- `tests/`：工程级 UT。

## 调用约定

安装后的 wheel 公开 Python import 面只使用 `fla_npu`。不要让新代码依赖顶层
`fla` 包；`fla/` 目录主要作为源码树内实现来源存在。

Ascend C 新代码优先使用稳定 Python 入口：

```python
from fla_npu.ops.ascendc import chunk_bwd_dv_local
```

Triton 算子同样使用 `fla_npu` 下的稳定入口：

```python
from fla_npu.ops.triton import chunk_local_cumsum
```

默认 Ascend C 调用路径必须保持与 `torch_npu` dispatcher、PyTorch C++ extension ABI、CPython ABI 和 C++ ABI 解耦。`fla_npu.ops.ascendc` 通过 Python ctypes 直调 `aclnn*`，不得在普通 import 或默认算子调用时 import `torch_npu`、注册 `torch.ops.npu`，或依赖 `custom_aclnn_extension_lib*.so`。

`torch.ops.npu.*` 是兼容旧调用的过渡路径，仅在兼容性测试或旧 API 验证中使用。需要旧路径时先确认 `fla_npu.load_legacy_torch_ops()` 的加载逻辑，并在 PR 中说明为什么不能使用 `fla_npu.ops.ascendc.<op>`。

修改 `torch_custom/fla_npu/fla_npu/ops/ascendc/_runtime.py`、`_aclnn_ctypes.py`、`torch_custom/fla_npu/setup.py` 或根目录 `setup.py` 时，必须同步检查 `docs/agents/architecture/torch-npu-decoupled.md`。涉及依赖确定阶段、版本或能力门禁、SOC/host/CANN 兼容范围、多卡 device guard、stream 感知、异步 launch 保活、正反向绑定、ACL 私有 format 透传、OPP wheel 安装位置或 legacy `torch_npu` 兼容路径的行为变化时，文档必须一起更新。

ctypes 算子如果会通过 data pointer 修改输入 tensor，必须在公共 wrapper 中显式维护 alias/mutation 契约：列出 mutated args，处理 eager autograd 版本计数，明确被修改状态的 grad 限制，并补充 mutation 测试。未增加 `torch.library` mutation schema、FakeTensor 和 `opcheck` 前，不得宣称该 mutable 路径支持 `torch.compile`、functionalization 或 `torch.export`；需要完整图编译支持时优先提供纯 Python custom-op 适配或返回新状态的 functional API，不得为此退回 PyTorch C++ extension。

## ctypes aclnn ABI 一致性红线

1. `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 中显式声明的 `GetWorkspaceSize` 参数类型，必须逐项对照对应 `aclnn*GetWorkspaceSize` 原型，严格保持输入、可选输入、属性、输出、`workspaceSize` 和 `executor` 的数量、顺序与 C 类型一致。每一项必须用行内注释标明对应参数名，禁止仅凭相邻算子、旧版本或参数总数推测。
2. wrapper 构造的实参数量和顺序必须与参数类型表去掉末尾 `workspaceSize`、`executor` 后完全一致。新增或修改显式参数类型时，必须补充不依赖 NPU 的 ABI 契约测试，同时断言完整类型序列、wrapper 实参数量和逐项 ctypes 类型；不能只依赖上板执行发现 `ctypes.ArgumentError`。
3. `_aclnn_ctypes.py`、`_runtime.py` 等共享适配文件中的无关算子条目不得随当前算子重构被删除、重排或改型。修改共享 ABI 表后必须检查聚焦 diff，并运行所有 ctypes ABI 契约测试；PR 的验证范围不能只覆盖当前功能算子。
4. 修改 aclnn C++ 原型时，必须在同一变更中同步 Python ctypes 类型表、wrapper 实参、schema、公开文档和 ABI 测试。仅修改 Python 适配但不改变公开 C++ 原型时，也必须明确说明 ABI 不变，并以对应原型作为修复依据。

## 算子开发交付 checklist

新增或修改 Ascend C 算子时，交付前执行 `docs/agents/development/checklist.md`。涉及性能设计或优化时同时执行 `docs/agents/optimization/checklist.md`。至少确认接口、host、kernel、schema、Python 导出、测试、文档和目标 SOC 保持一致。

ABI 敏感路径包括 `*_def.cpp`、`aclnn_*.h/.cpp`、`torch_custom/fla_npu/*.yaml` 和 `torch_custom/fla_npu/op_plugin/ops/opapi/**`。修改这些文件时，PR 需要明确说明 ABI 影响，并按 `.github/CODEOWNERS` 请求对应 owner 检视。

## 构建命令

完整的环境、构建、安装和调用方法以根目录 `README.md` 为准。开始前先执行环境检查：

```sh
python scripts/check_npu_env.py --build-only
```

源码或 Python 适配修改后使用目标 SOC 完整重编一体化 wheel：

```sh
FLA_NPU_SOC=<soc> python -m pip wheel --no-build-isolation --no-deps . -w dist
```

SOC 映射：A2 为 `ascend910b`，A3 为 `ascend910_93`，A5 为 `ascend950`。单算子 run 包只用于定位，不能替代完整 wheel 重编。

## 安装和验证

验证方法和矩阵见 `docs/agents/development/validation.md`。安装 wheel 后至少检查公开 API：

```sh
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
python scripts/check_packaged_wheel_api.py
```

单算子、Example/ST 和 CI 命令从 `README.md`、当前算子 README 和现有脚本选择。缺少 CANN、NPU 或运行时依赖时，不得把未执行命令写成已验证结论。

## 测试要求

- 修改参数校验、shape、dtype、layout、range、平台差异或预留参数语义时，补充反向测试和边界测试。
- 修改 Kernel 时至少覆盖对应单算子测试；涉及 GDN 端到端链路时跑 Example/ST。
- 修改 ABI、公共模块或共享路径时扩大回归范围，至少说明影响到的算子。
- 精度失败不能通过收窄输入 range、跳过 case、降低覆盖强度或放宽阈值来制造通过结论；应先定位误差来源，再修 kernel、标杆或后处理语义。
- 性能结论以合适的 profiling/CI 结果为准，不用 Python wall time 直接下结论。

## 构建产物与提交规范

不要提交构建、安装、调测和性能分析产物。重点检查：

- `build/`、`build_out/`、`output/`、`dist/`、`.ci-cache/`、`third_party/`
- `torch_custom/fla_npu/build/`、`torch_custom/fla_npu/dist/`、`torch_custom/fla_npu/torch_npu/`
- `torch_custom/fla_npu/test/test_output/`、`torch_custom/fla_npu/test/data/`
- `__pycache__/`、`*.pyc`
- `.tmp*`、`outputs/`、`PROF_*`、`OPPROF_*`、`extra-info`

提交前至少运行：

```sh
git status --short
git diff --check
```

## PR 和 CI

- PR 描述使用 `.github/pull_request_template.md`，不要自创栏目替代模板。
- PR 应关联 Issue，或在模板中说明无需 Issue 的原因。
- NPU CI 不会在每次 push 后自动跑；需要仓库 Admin 在 Actions 手动触发，或在 PR 评论 `/run-npu-ci quick` / `/run-npu-ci full`。
- 当前 head commit 需要通过 `NPU CI / 手动验证` 和 `NPU CI / 精度检查`，并满足 2 个 approval。
- push 新 commit 后旧 commit 的 CI 结果失效，需要重新触发。

## 给 AI 的最后检查

结束任务前确认：

- 改动是否只覆盖本次任务需要的文件。
- 文档、报错、返回码、schema、Python 导出和测试是否与代码语义一致。
- 是否遗漏对应 SOC、layout、dtype、varlen/dense、边界 case。
- 是否有未跟踪生成物或敏感信息混入。
- 是否清楚说明已执行和未执行的验证。
