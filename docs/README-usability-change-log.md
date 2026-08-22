# README 易用性整改 · 修改点记录（21 点全量）

> 协作方式：每确认一个修改点 → 记入本表 → 确认下一个 → 最后一次性实施。
> 优先级说明：P0 = 事实性错误 / 会让用户操作失败；P1 = 核心需求缺失；P2 = 易用性增强。
> **状态总览（2026-08-08 17:00）**：21 点已全部实施并提交；PR #280 评审意见 10 条已修订（G 轮）；复审 3 条已修订（H 轮）；行内【review】评论 5 条已修订（I 轮）；预检相关 2 条已修订（J 轮）；新增 4 条已修订（K 轮）；新增 2 条已修订（L 轮）。分支 `20260806_204500_docs-README-usability`（基于最新 main `ac46f1c3`，含 PR #274），已推送至 origin（Coding-Pangolin）。
> 口径：PR #274 已合入 main（`ac46f1c3`）。下文凡涉及"增量构建移除、`import fla_npu` 即加载、`scripts/check_install_workflows.py`、卸载说明、shell 环境钩子"等，均为 #274 在 main 上已有的内容；本次 PR 的增量是文档修正与新章节，与 #274 语义兼容（冲突合并时保留 #274 侧内容 + 叠加本次修正）。

---

## A. 根 README.md

### 修改点 #1（P0）：修正 `check_npu_env.py --build-only` 描述并补充完整预检方法（含 A1）

- **状态**：已实施
- **涉及**：`README.md` Step 2（原"如果依赖缺失，预检和一键编包都会在真正编译前失败，并列出缺失项……"）
- **依据（实测）**：`python scripts/check_npu_env.py --build-only` 在缺 torch 时也通过（跳过 torch/torch_npu/torchnpugen/triton 检查，`EXIT=0`），原文描述与行为不符；且原文档未告知如何判断 torch 系依赖是否完整、版本是否匹配。
- **实际改法**：明确 `--build-only` 只检查构建纯 Python wheel 所需环境（Python / bash / CANN），不检查 torch 系依赖；新增完整预检命令 `python scripts/check_npu_env.py`，说明其检查 `torch` / `torch_npu` / `triton-ascend` 是否可导入、版本下限与 NPU 可用性（`torch.npu.is_available()`），并说明无 NPU 卡的纯构建环境 `is_available()` 为 `False` 属预期、此时用 `--build-only` 即可；torch 系依赖缺失或版本不匹配时 `pip wheel` 会在构建/打包阶段报错，需按 CANN 与 Python 版本匹配的列表先行安装。

### 修改点 #2（P0）：移除已废弃的增量构建描述

- **状态**：已实施（后续依据用户结论回滚 `FLA_NPU_OPS` 保留，见修改点 #2 补充）
- **涉及**：`README.md` Step 2 方式 A（原"真增量构建"段落 + 环境变量表 `FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_OPS` 两行）
- **依据**：PR #274 移除增量构建开关，改为全量重建；只定位单算子用 `bash build.sh --ops=<op>`。
- **实际改法**：环境变量表删除 `FLA_NPU_INCREMENTAL_BUILD`、`FLA_NPU_OPS` 两行。PR #274 在 setup.py 中**一并移除了 `FLA_NPU_SKIP_RUN_BUILD` / `FLA_NPU_SKIP_RUN_INSTALL`**（`tests/test_wheel_environment.py` 有守卫断言二者不得出现在 setup.py），因此这两行也随 #274 删除；最终环境变量表仅保留 `FLA_NPU_SOC` / `FLA_NPU_DISABLE_LOCAL_VERSION`（与 ac46 一致）。合并冲突时，"全量构建"段落采用 #274 在 main 上已有的版本（含"清理 `build/`/`build_out/`/`output/` 中间产物"及"dist 下可能多版本 wheel、需用准确文件名"的提示），不再自行改写。
- **修改点 #2 补充（后续回滚）**：经实测确认 PR #274 移除了 `FLA_NPU_OPS` 属于错误移除（该变量支持"已装完整 wheel 后快速替换少量算子 Ascend C 产物"的单算子 wheel 构建），应保留。已在 `setup.py` `_build_run_package()` 恢复 `FLA_NPU_OPS` 读取并透传 `build.sh --ops=<op>`；README 环境变量表加回 `FLA_NPU_OPS` 行；`tests/test_wheel_environment.py` 守卫由 `assertNotIn("FLA_NPU_OPS")` 改为新增 `test_package_build_supports_single_op_filter` 断言其存在；`FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_SKIP_RUN_BUILD` / `FLA_NPU_SKIP_RUN_INSTALL` 维持移除。

### 修改点 #3（P0）：修正 Step 4 验证命令，区分新旧 wheel 行为（含 A2）

- **状态**：已实施
- **涉及**：`README.md` Step 4（原 `is_legacy_torch_ops_loaded()` + `hasattr(torch_npu.ops, 'chunk_fwd_o')` 验证）
- **依据（实测）**：`torch_npu.ops` 兼容入口行为随 wheel 版本而异——**当前仓库源码（PR #274 后）** 下 `torch_npu.ops` 属性不存在，旧命令抛 `AttributeError`；**fzy 安装的旧 wheel（26.7.0.dev0，2026-07-13 构建）** 导入 `fla_npu.ops.ascendc` 时自动调用 `install_torch_npu_ops_compat()`，`hasattr` 返回 `True`。
- **实际改法**：验证命令改为 `python -c "import fla_npu; print('ok')"` + `python scripts/check_packaged_wheel_api.py`；Step 4 新增两个 bullet 区分新旧 wheel 行为：新版默认不注册 `torch_npu.ops.*`，旧命令抛 `AttributeError` 属预期行为、不要用它验证新版，需要时先显式调用 `install_torch_npu_ops_compat()`；旧版（2026-07 之前的中间版本）导入即自动挂载、`hasattr` 返回 `True`，属旧版行为、不代表新 wheel。**实测补充（2026-08-07）**：`install_torch_npu_ops_compat()` 的调用必须先导入子模块——`import fla_npu` 后直接写 `fla_npu.ops.ascendc.install_torch_npu_ops_compat()` 会报 `AttributeError: module 'fla_npu' has no attribute 'ops'`（顶层 `__init__.py` 不自动导入 `ops`），必须 `from fla_npu.ops import ascendc`（或 `import fla_npu.ops.ascendc`）后再调用；已按此修正 Step 4 说明，给出可复制的 Python 片段。合并冲突时，Step 4 同时保留 #274 已加入的"卸载说明"（`pip uninstall flash-linear-attention-npu` 与 RECORD 无残留说明）及 `scripts/check_install_workflows.py` 看护脚本用法，与本次修正的验证命令共存。

### 修改点 #4（P1）：安装命令使用精确文件名并强制覆盖同版本号

- **状态**：已实施
- **涉及**：`README.md` Step 3 方式 A 与方式 B（原通配符 `dist/flash_linear_attention_npu-*.whl`）
- **依据**：通配符可能匹配多个 wheel 产物导致安装失败；重新构建 wheel 的版本号可能与已安装旧 wheel 相同（如本地 dev 版本 `26.7.0.dev0`），不带 `--force-reinstall` 的 `pip install` 会认为"已是最新版本"而跳过，导致实际仍是旧代码。
- **实际改法**：改为 `WHEEL_PATH="dist/<准确wheel文件名>.whl"` + `python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"`；方式 B 同理（`torch_custom/fla_npu/dist/<准确wheel文件名>.whl`）。新增引用块说明：版本号相同时需 `--force-reinstall` 强制覆盖，或先 `python -m pip uninstall -y flash-linear-attention-npu` 再装。

### 修改点 #5（P1）：`set_env.sh` 保留默认路径 + 补充自定义路径提示

- **状态**：已实施
- **涉及**：`README.md` Step 1（`export INSTALL_PATH` 注释 + `source $INSTALL_PATH/ascend-toolkit/set_env.sh` 后）
- **用户原话**：保留 `/usr/local/Ascend/ascend-toolkit/set_env.sh` 作为默认，补充"若安装在自定义路径请 source 实际路径下对应的 set_env.sh"，但**文档中不出现具体自定义路径示例**。
- **实际改法**：Step 1 命令注释改为"设置需要安装的路径（请替换为实际安装路径）"，代码块后新增引用块：自定义路径时设置 `INSTALL_PATH` 为实际安装路径并 source 对应 `set_env.sh`；每次进入新 shell（Docker/Conda/venv）需重新 source。

### 修改点 #6（P1）：新增"从旧版本升级"章节，并补充新旧版本行为差异（含 B1/B2/B3）

- **状态**：已实施
- **涉及**：`README.md` 新增 `## 开发者指引 > ### 从旧版本升级（v26.6.0 及更早 → 最新）`（置于 Step 4 之后）
- **依据**：用户需求 #2/#3——`torch.ops.npu.*` 只支持到 v26.6.0，后续需 `fla_npu.ops.ascendc`；升级用户需要知道旧版与新版的构建、验证、兼容入口行为差异。
- **实际改法**：5 步——卸载旧包并清理残留（含 `custom_aclnn_extension_lib*.so` / 自定义 `libopapi.so`）→ 安装新版本 wheel → 迁移调用（旧→新对照表：`torch.ops.npu.npu_chunk_fwd_o` → `from fla_npu.ops.ascendc import chunk_fwd_o` 等）→ 验证（`check_packaged_wheel_api.py` + `test.sh --op gdn_fwd_o`）→ 迁移期临时兼容（`install_torch_npu_ops_compat()` / `load_legacy_torch_ops()`，注明 legacy 需 `FLA_NPU_BUILD_LEGACY_EXTENSION=1`，新代码勿用 legacy；**实测补充（2026-08-07）**：调用 `fla_npu.ops.ascendc.install_torch_npu_ops_compat()` 前必须先 `from fla_npu.ops import ascendc`，`import fla_npu` 后直接全限定名调用会报 `AttributeError`，已修正描述）。章节末尾新增"旧版本（≤ v26.6.0）与新版的主要行为差异"四项：
  - **B1 构建环境变量**：旧版支持 `FLA_NPU_INCREMENTAL_BUILD`（增量构建）、`FLA_NPU_OPS`（单算子 wheel）、`FLA_NPU_SKIP_RUN_BUILD` / `FLA_NPU_SKIP_RUN_INSTALL`（run 包控制）；新版（PR #274 起）移除了 `FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_SKIP_RUN_BUILD` / `FLA_NPU_SKIP_RUN_INSTALL`，统一全量构建（自动清理 `build/`/`build_out/`/`output/` 中间产物），单算子定位改用 `bash build.sh --pkg --soc=<soc> --vendor_name=fla_npu --ops=<op>` 构建 run 包，旧脚本中的 `FLA_NPU_INCREMENTAL_BUILD=1` 需要删除；`FLA_NPU_OPS`（单算子 wheel）经确认 #274 属错误移除，已恢复保留（`setup.py` 透传 `build.sh --ops=<op>`）。
  - **B2 验证方式**：旧版 Step 4 的 `fla_npu.is_legacy_torch_ops_loaded()` 与 `hasattr(torch_npu.ops, ...)` 在新版不再适用；统一改用 `python -c "import fla_npu; print('ok')"` + `python scripts/check_packaged_wheel_api.py`。
  - **B3 `torch_npu.ops` 挂载行为**：旧版 wheel（2026-07 之前的中间版本）导入 `fla_npu.ops.ascendc` 即自动挂载 `torch_npu.ops.*`；新版（PR #274 后构建）默认不挂载，迁移期需显式调用 `install_torch_npu_ops_compat()`。
  - **`test.sh` 算子名**：`recompute_wu_fwd` 在新版统一为 `recompute_w_u_fwd`。

### 修改点 #7（P1）：新增"在 torch_custom 新增 Python 接口"指引

- **状态**：已实施
- **涉及**：`README.md` 新增 `### 在 torch_custom 新增 Python 接口`（位于升级章节后）
- **依据**：用户需求 #3——torch_custom 下怎么加新接口。
- **实际改法**：4 步核心链路——`_aclnn_ctypes.py` 新增 `npu_xxx(...)` wrapper → `__init__.py` 的 `_ASCENDC_OPS` 注册（自动导出 `npu_xxx` 与去前缀短名）→ 新增 `test_npu_<op>.py` 并接入 `test.sh` → 重新构建 wheel/run 包并验证；链接到 `torch_custom/fla_npu/README.md` 详情。

### 修改点 #8（P1）：新增"全新环境快速上手"前置步骤与冒烟命令

- **状态**：已实施
- **涉及**：`README.md` 新增 `Step 0. 确认硬件与目标芯片`（Step 1 之前）；Step 4 末尾补冒烟测试
- **依据**：用户需求 #1——全新环境按文档能否跑通。
- **实际改法**：Step 0 用 `npu-smi info` 确认机器类型 + A2/A3/A5 ↔ `--soc`/`FLA_NPU_SOC` 对照表；Step 4 末尾补真实算子冒烟 `cd torch_custom/fla_npu/test && bash test.sh --device 0 --op gdn_fwd_o`。

### 修改点 #9（P2）：补全 `test.sh --op` 可选值

- **状态**：已实施
- **涉及**：`README.md` "测试单算子"可选值列表
- **依据**：实测 `test.sh` 含 `chunk_local_cumsum`、`chunk_scaled_dot_kkt` 两个已接入任务，原列表缺失。
- **实际改法**：列表补上 `chunk_local_cumsum`、`chunk_scaled_dot_kkt`，共 11 个，与 `test.sh` 逐条核对一致。

### 修改点 #10（P2）：精简方式 B 的 legacy 编译命令

- **状态**：已实施
- **涉及**：`README.md` Step 2 方式 B（原含 `FLA_NPU_BUILD_LEGACY_EXTENSION=1 bash gen.sh npu_custom.yaml` 与 `setup.py bdist_wheel` 两条）
- **依据**：legacy extension 是迁移期可选能力，放在主路径会让使用者误以为必须编译。
- **实际改法**：从方式 B 删除这两条命令；legacy 说明保留在"从旧版本升级"章节的迁移期临时兼容条目中。

### 修改点 #11（P2）：修正 CANN 下载段文案

- **状态**：已实施
- **涉及**：`README.md` Step 1（原"推荐使用是社区版8.5.2，总共要下2个run包"）
- **依据**：原句语法错误；A2/A3/A5 需下载对应 ops 与 toolkit 包。
- **实际改法**：改为"推荐社区版 8.5.2，总共需要下载 2 个 run 包。这里以 A3 机器为例（即需要下载 A3-ops 与 toolkit），A2 / A5 机器请下载对应的 ops 与 toolkit 包。"

### 修改点 #12（P2）：概述补充依赖版本匹配说明

- **状态**：已实施
- **涉及**：`README.md` 概述段末尾
- **依据**：本仓不自动安装 torch 系依赖，需明确版本匹配要求。
- **实际改法**：新增一段——"本仓不自动安装 `torch`、`torch_npu`、`torchnpugen`、`triton-ascend`，这些包必须与 CANN 与 Python 版本匹配，需要使用者按环境自行安装；版本不匹配时，构建或运行会报错。"

---

## B. torch_custom/fla_npu/README.md

### 修改点 #13（P1）：新增"导入契约"章节

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` 开头（默认交付目标段后）
- **依据**：`import fla_npu` 即定位并加载 `libcust_opapi.so`，用户需要知道导入前提与常见失败原因。
- **实际改法**：新增章节说明——导入/构建前先 source CANN `set_env.sh`（默认路径 + 自定义路径提示）；以表格列出常见现象与处理：standalone wheel 缺 OPP（需先装 run 包或设 `FLA_NPU_OPP_PATH`）、未 source set_env.sh 导致 dlopen 报错、安装 run 包后需重启 Python 进程。

### 修改点 #14（P1）：修正"新算子如何接入默认 runtime"

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` "新算子如何接入默认 runtime"步骤 6/7
- **依据**：原步骤缺少 mutation 契约与正反向绑定细节，`MUTATED_ARGUMENTS` / `BACKWARD_OPS` 实际存在于 `__init__.py`。
- **实际改法**：步骤 6 明确 `BACKWARD_OPS` 映射（例 `causal_conv1d` → `causal_conv1d_bwd`）与 autograd 自动绑定；新增步骤 7 说明就地修改输入 tensor 的算子需在 `MUTATED_ARGUMENTS` 登记参数名（ctypes 直写 storage 时 PyTorch 无法自动发现副作用），并指向 `test/test_ascendc_mutation_contract.py`；原步骤 7 顺延为 8。

### 修改点 #15（P2）：构建/安装命令对齐 PR274

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` "构建和验证默认 runtime" 三处安装命令
- **依据**：与根 README 修改点 #4 一致；`scripts/check_install_workflows.py` 已随 PR #274 合入 main。
- **实际改法**：三处 wheel 安装均改为 `--force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"`（`WHEEL_PATH` 用准确文件名）；在"构建和验证默认 runtime"末尾补充 `scripts/check_install_workflows.py` 使用说明（安装流程看护，CI 自动运行，标注随 PR #274 引入）。

### 修改点 #16（P2）：legacy 章节补充迁移指引

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` "legacy torch_npu / torch.ops.npu 路径" 章节末尾
- **依据**：与根 README 升级章节呼应，明确 legacy 支持边界。
- **实际改法**：末尾新增——"`torch.ops.npu.*` / `torch_npu.ops.*` 只支持到 v26.6.0，从旧版本迁移到最新版本的完整步骤见根 README 的'从旧版本升级'章节；新代码请勿使用 legacy 路径。"

---

## C. AGENTS.md

### 修改点 #17（P2）：移除增量构建命令并修复多余代码围栏（含 A3）

- **状态**：已实施（后续依据用户结论回滚部分，见修改点 #17 补充）
- **涉及**：`AGENTS.md` "构建命令"（原 `FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_OPS` 两条命令）
- **依据**：与根 README 修改点 #2 一致（PR #274 已合入 main）；实测发现"单算子 run 包命令"代码块后多出一个 ``` 围栏，导致后续 markdown 渲染异常。
- **实际改法**：删除两条增量/单算子 wheel 构建命令，改为"源码或适配修改后仍执行完整 wheel 构建；构建流程会清理上一轮 `build/`、`build_out/`、`output/` 中间产物，不再支持增量构建。只定位单算子时，用 `bash build.sh --pkg --soc=<soc> --vendor_name=fla_npu --ops=<op>` 构建单算子 run 包"（单算子产物不能替代完整 wheel 的全量重编），并给出示例命令；同时删除构建命令后多余的 ``` 代码围栏，修复 markdown 渲染。
- **修改点 #17 补充（后续回滚）**：因 `FLA_NPU_OPS` 判定为错误移除并恢复保留，`AGENTS.md` 在"完整 wheel 构建"命令后补回 `FLA_NPU_OPS` 单算子 wheel 构建示例（`FLA_NPU_OPS=<op> python -m pip wheel ...`）。

### 修改点 #18（P2）：`set_env.sh` 硬编码补自定义路径提示

- **状态**：已实施
- **涉及**：`AGENTS.md` "构建命令"环境准备段
- **依据**：与根 README 修改点 #5 一致。
- **实际改法**：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 上行加注释"CANN 安装于自定义路径时，请替换为实际路径下对应的 set_env.sh"。

---

## D. examples/README.md

### 修改点 #19（P2）：重写为实际样例

- **状态**：已实施
- **涉及**：`examples/README.md` 全文
- **依据**：原文件为 CANN ops-transformer 模板——引用不存在的 `mc2/`、失效链接 `../docs/zh/develop/aicore_develop_guide.md`，与本仓实际样例脱节。
- **实际改法**：重写为——简介（三个实际样例：`flash_gated_delta_rule.py`、`add_example/`、`fast_kernel_launch_example/`）→ 目录说明 → 快速运行（GDN 端到端、add_example、fast_kernel_launch_example）→ 新增示例要求（独立可运行、配单算子测试并接入 test.sh、优先 `fla_npu.ops.ascendc` / `.triton` 稳定入口、提供 CMakeLists.txt 参与统一编译）。

---

## E. CONTRIBUTING.md

### 修改点 #20（P2）：补充"本仓特有的贡献要求"

- **状态**：已实施
- **涉及**：`CONTRIBUTING.md` "贡献新算子"第 4 步之后
- **依据**：用户需求 #3/#4——新增算子必须提供稳定 Python 入口。
- **实际改法**：新增章节——新增算子必须提供 `fla_npu.ops.ascendc` 稳定 Python 入口，不得仅以 legacy `torch.ops.npu.*` / `torch_npu.ops.*` 交付；交付内容含 `_aclnn_ctypes.py` wrapper、`_ASCENDC_OPS` 注册（需要时同步 `BACKWARD_OPS` / `MUTATED_ARGUMENTS`）、`test_npu_<op>.py` 并接入 `test.sh`。

---

## F. .github/pull_request_template.md

### 修改点 #21（P2）：`set_env.sh` 硬编码补自定义路径提示

- **状态**：已实施
- **涉及**：`.github/pull_request_template.md` 验证方法 > 环境确认
- **依据**：与根 README 修改点 #5 一致。
- **实际改法**：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 上行加注释"默认 CANN 安装路径；自定义安装路径时替换为实际路径下对应的 set_env.sh"。

---

## 实施后可实测项验证结果

| 验证项                                                                                    | 结果                                                                                                                                                      |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python scripts/check_npu_env.py --build-only`（缺 torch）                              | 通过，`EXIT=0`，跳过 torch 系检查（印证 #1）                                                                                                            |
| 完整预检`python scripts/check_npu_env.py`（fzy 环境）                                   | 可检查 torch/torch_npu/triton-ascend 与`is_available()`（#1）                                                                                           |
| Step 4 新旧 wheel 行为（fzy 实测）                                                        | 旧 wheel（26.7.0.dev0）`hasattr` 返回 `True`；新版源码行为为抛 `AttributeError`（#3）                                                               |
| `bash test.sh --device 0 --op gdn_fwd_o --mode dry-run`                                 | 正常打印命令（印证#8/#9）                                                                                                                                 |
| `bash build.sh --help`                                                                  | `--pkg/--soc/--ops/--vendor_name` 参数存在                                                                                                              |
| `scripts/check_install_workflows.py`                                                    | 已随 PR#274 合入 main（修改点 #15 引用了其用法）                                                                                                          |
| `FLA_NPU_OPP_PATH` 环境变量                                                             | `fla_npu/__init__.py`、`install_opp.py` 中真实支持（修改点 #13）                                                                                      |
| `install_torch_npu_ops_compat()` / `load_legacy_torch_ops()`                          | `fla_npu/ops/ascendc/__init__.py`、`fla_npu/__init__.py` 中真实存在（#3/#6）                                                                          |
| `install_torch_npu_ops_compat()` 调用方式（fzy 实测）                                   | `from fla_npu.ops import ascendc` 后调用可挂载 `torch_npu.ops`；`import fla_npu` 后全限定名调用报 `AttributeError`，已按正确写法修正 Step 4（#3） |
| `_GET_WORKSPACE_ARGTYPES` / `BACKWARD_OPS` / `MUTATED_ARGUMENTS`                    | 代码中真实存在（修改点#14）                                                                                                                               |
| AGENTS.md 代码围栏修复                                                                    | `git diff` 确认删除多余 ```（#17）                                                                                                                      |
| 新增文档链接目标                                                                          | 全部有效                                                                                                                                                  |
| 残留校验（`FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_SKIP_RUN_BUILD` / `FLA_NPU_SKIP_RUN_INSTALL` / `mc2` / `../docs/zh` 等） | 本次改动文档中无残留；`FLA_NPU_OPS` 为恢复保留项（见修改点 #2/#17 补充） |
| `git diff --check`                                                                      | 通过                                                                                                                                                      |
| PR 冲突检查（compare API）                                                                | 分支基于`ac46f1c3`（main），无冲突                                                                                                                      |

**未实测**：需要真实 NPU + torch 环境才能执行的算子冒烟（`bash test.sh --device 0 --op gdn_fwd_o` 实际执行），本机缺 torch 环境，未执行。

---

## G. PR #280 评审意见修订（2026-08-07）

PR #280 上 reviewer 共提出 10 条行内意见，已逐条修订并登记如下。

### 意见 1/2（README Step 1）：CANN 版本推荐改最新稳定版 + 稳定链接

- **原状**：Step 1 写死"推荐社区版 8.5.2"，下载链接指向 8.5.2 具体版本页。
- **意见**：① 要求 8.5.2 以后版本，推荐使用最新社区稳定版本；② 更换成稳定版本链接。
- **实际改法**：文案改为"推荐使用最新的社区稳定版本（不低于 8.5.2，如需使用更新版本请参考 `check_npu_env.py` 支持的 CANN / torch_npu 版本组合）"；下载链接改为社区 CANN 下载总入口 `https://www.hiascend.com/developer/download/community`，并在其中选择最新的稳定版本。**第二轮修订（2026-08-07）**：按 reviewer 提供的稳定版链接，改为 `https://www.hiascend.com/zh/cann/download?versionId=752&ids=d803%2Ch0501%2Ch0601%2Ch0703`（已实测可访问，为最新稳定版下载页）。

### 意见 3（README Step 1）：修正 A3 写死的问题

- **原状**：安装命令写死 `./Ascend-cann-A3*run`，A2/A5 用户照抄会失败。
- **意见**：A3 写死的需要修正。
- **实际改法**：改为 `./Ascend-cann-<机器型号>*run`，注释说明"toolkit 与机型对应的 ops 包都必须安装，`<机器型号>` 请替换为实际机型对应的包前缀"，并给出 A3 → `Ascend-cann-A3*run`、A2 → `Ascend-cann-910b*run`、A5 → `Ascend-cann-950*run` 示例（已对照昇腾社区 CANN 官方 run 包命名核实）。**第二轮修订（2026-08-07）**：按 reviewer 意见改为 ops 通配符——ops 包统一用 `./Ascend-cann-*-ops*.run`（社区 ops 包命名格式 `Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run`，如 `Ascend-cann-A3-ops*.run` / `Ascend-cann-910b-ops*.run` / `Ascend-cann-950-ops*.run`），用户无需再手动替换机器型号。

### 意见 4（README Step 2）：`--build-only` 环境不全补充 + 优先推荐完整预检

- **原状**：仅说明 `--build-only` 不检查 torch 系依赖。
- **意见**：① `--build-only` 检查的环境可能不全（CMake 等组件未检查），通过也可能不能正常编译，需补充支持的版本；② 优先建议用户用同时检测执行与编译的（不带 `--build-only`）。
- **实际改法**：改为"**建议优先使用完整预检（不带 `--build-only`）**"开头；补充说明 `--build-only` 不覆盖 `CMake`、编译器（`g++` / `bisheng`）、Python 头文件、`ninja` 等编译组件，检查通过不代表一定可以编译，缺失的编译组件会在 `pip wheel` 阶段才暴露。

### 意见 5（README Step 4）：去掉 PR 号，改描述 v26.6.0 后不维护兼容接口

- **原状**：Step 4 兼容入口说明写"新版 wheel（从当前仓库源码构建，PR #274 之后）"。
- **意见**：不应该显示写 PR 号，增加描述 v26.6.0 后不维护旧版本的兼容接口。
- **实际改法**：README 正文不再出现 PR 号；Step 4 改为"`torch.ops.npu.*` / `torch_npu.ops.*` 是旧版本（v26.6.0 及更早）的调用方式，**v26.6.0 之后不再维护旧版本兼容接口**，新代码请使用 `fla_npu.ops.ascendc` 下的稳定 Python 入口"，兼容细节（`install_torch_npu_ops_compat()` / `load_legacy_torch_ops()` 用法、`hasattr` 版本差异）移入新文档。

### 意见 6/10（README Step 4）：兼容性内容拆分到独立文档

- **原状**：Step 4 用大段篇幅说明 `torch_npu.ops.*` 兼容入口的新旧行为差异；方案 B / legacy 说明分散在主流程。
- **意见**：① README 主要写主流程，兼容性的篇幅太大，单起一个文档更合适；② 方案 B 是 legacy 路径，legacy 路径都单独放，引导用户用新方式。
- **实际改法**：新建 `docs/migration-guide.md`（兼容与迁移指南），收纳：调用方式演进对照表、从旧版本升级完整步骤、迁移期临时兼容（`install_torch_npu_ops_compat()` / `load_legacy_torch_ops()` 及 legacy 构建命令）、新旧版本行为差异、`hasattr(torch_npu.ops, ...)` 版本差异注意事项。README Step 4 与"开发者指引"只保留主流程 + 指向该文档的链接。

### 意见 7（README Step 4）：冒烟测试用算子名

- **原状**：冒烟测试 `bash test.sh --device 0 --op gdn_fwd_o` 未说明 `--op` 参数含义。
- **意见**：冒烟测试可以保留，但应该用算子名。
- **实际改法**：补充说明"`--op` 后跟算子名，`gdn_fwd_o` 为示例"。

### 意见 8（README 开发者指引）：开发者部分分场景拆分 + 单起 md

- **原状**："开发者指引"下按文档章节堆叠：从旧版本升级、在 torch_custom 新增接口、测试单算子、算子调用方式参考、端到端验证。
- **意见**：开发者下面分为多个场景区分——单独编译单算子（`bash build.sh` 方式、一键编包单算子）、增加一个算子的方式（目录结构、torch_custom），尽量对开发者透明，单起一个 md。
- **实际改法**：新建 `docs/developer-guide.md`（开发者指南），按场景拆分：场景 1 单独编译单算子（run 包）、场景 2 一键编包单算子、场景 3 增加一个新算子（目录结构 + torch_custom）、场景 4 测试单算子、场景 5 端到端 Example/ST 验证。README"开发者指引"精简为场景导航列表 + 链接。

### 意见 9（README 开发者指引）：如何确认新编版本来自最新源码

- **原状**：未说明如何确认新编的 wheel 确实由最新修改的源码编译。
- **意见**：怎么确定新编的版本是最新修改的源码编译的（比如开 debug 日志确定走的版本），需要找个方案。
- **实际改法**：在 `docs/developer-guide.md` 新增"如何确认新构建的 wheel 来自最新源码"章节，给出三种方式：方式 1 核对 wheel 文件名与版本号（构建日志文件名 → `pip show` / `importlib.metadata` 核对）；方式 2 确认实际加载的 OPP 路径（`import fla_npu` 后打印 `fla_npu.__file__` 与 `FLA_NPU_OP_API_LIB` / `ASCEND_CUSTOM_OPP_PATH` 环境变量，确认指向新 wheel 内嵌 OPP）；方式 3 修改后强制覆盖安装（`--force-reinstall`，或临时打印标记确认后移除）。

---

## H. 第二轮修订（2026-08-07）

PR #280 复审后 reviewer 追加 3 条要求，已修订并登记如下。

### 修订 H1：稳定版链接按 reviewer 提供链接替换

- **原状**：Step 1 下载链接为社区下载总入口 `https://www.hiascend.com/developer/download/community`。
- **意见**：reviewer 在行内评论给出社区稳定版具体链接。
- **实际改法**：替换为 reviewer 提供的 `https://www.hiascend.com/zh/cann/download?versionId=752&ids=d803%2Ch0501%2Ch0601%2Ch0703`（已实测可访问），文案标注"最新稳定版"。同步更新意见 1/2 登记。

### 修订 H2：A3 写死改为 ops 通配符

- **原状**：安装命令为 `./Ascend-cann-<机器型号>*run`，仍需用户手动替换机型前缀。
- **意见**：改成 ops 的通配符，参考社区上包的格式。
- **实际改法**：ops 包改为 `./Ascend-cann-*-ops*.run`（社区 ops 包命名格式 `Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run`），注释给出 A3/A2/A5 对应 `Ascend-cann-A3-ops*.run` / `Ascend-cann-910b-ops*.run` / `Ascend-cann-950-ops*.run` 示例，用户无需手动替换机器型号。同步更新意见 3 登记。

### 修订 H3：开发者指引在 README 只留入口，场景导航并入独立文档

- **原状**：README"开发者指引"章节仍保留 6 条场景导航列表（单独编译单算子 / 一键编包 / 增加新算子 / 测试单算子 / 端到端验证 / 确认 wheel 来源）。
- **意见**：开发者的也拆分出一个文档。
- **实际改法**：README"开发者指引"精简为一行入口链接指向 `docs/developer-guide.md`；在 `docs/developer-guide.md` 开头补充场景目录（6 个锚点链接），独立文档自含完整导航，与迁移指南并列于 README"维护文档"。

---

## I. 第三轮修订（2026-08-08，PR #280 行内【review】评论 5 条）

reviewer 在 README 中新增 5 条【review】行内评论，已最小化修订并登记如下。

### 修订 I1（README Step 1 代码块注释）：换行调整

- **评论**：`README.md:46`「这里的换行有点怪，把例如A3机器也放到下一行更合适」。
- **实际改法**：注释第 2 行在"…`_linux-<arch>.run`，"处断行，"例如 A3 机器对应 …"整体放到第 3 行。

### 修订 I2（README Step 1 文案）：删去 A3 举例描述

- **评论**：`README.md:36`「这里已经没有以A3机器举例了，举例在后面，可以删去这部分描述」。
- **实际改法**：Step 1 首段删去"这里以 A3 机器为例（即需要下载 A3-ops 与 toolkit），A2 / A5 机器请下载对应的 ops 与 toolkit 包。"，机型对应关系由代码块注释统一给出。

### 修订 I3（README Step 3 说明）：删去本地 dev 版本示例

- **评论**：`README.md:127`「这里的如本地dev版本不合适，删去」。
- **实际改法**：同版本号提示删去"（如本地 dev 版本 `26.7.0.dev0`）"。

### 修订 I4（README Step 4 冒烟测试）：示例换成更准确的算子测试

- **评论**：`README.md:181`「这个地方举例更换一个准确算子的测试」。
- **实际改法**：冒烟测试示例由 `gdn_fwd_o`（走 `bash run_gdn_fwd_o.sh` 封装）改为 `causal_conv1d`（`python3 test_npu_causal_conv1d.py` 直接单算子测试，与 developer-guide 场景 4 示例一致）。

### 修订 I5（README Step 4 / 开发者指引）：测试与端到端验证归属

- **评论**：`README.md:223`「这里测试单算子和端到端测试需要确定下放哪里更合适」。
- **实际改法**：确定归属为"使用者快速验证放主流程、开发者全量测试放开发者文档"——README Step 4 冒烟测试后新增"可选的端到端验证"小节（`python examples/flash_gated_delta_rule.py`）；README"开发者指引"场景描述去掉"测试单算子、端到端验证"，改为注明"测试单算子和端到端验证见上文 Step 4"；`docs/developer-guide.md` 场景 4/5 开头补充"使用者快速验证见根 README Step 4，本场景面向开发调试/新增用例"。

---

## J. 第四轮修订（2026-08-08，PR #280 行内【review】评论 2 条）

reviewer 在 Step 2 预检处新增 2 条【review】评论（同一处），已修订并登记如下。

### 修订 J1（README Step 2）：预检命令直接用完整预检，`--build-only` 作为补充场景

- **评论**：`README.md:66`「这里虽然建议优先使用，但是上面给的操作还是带了--build_only，我的意思是直接在给出的命令中就指明直接用完整预检，而不是文字描述，然后后面再补充使用--build_only适用的情况」。
- **实际改法**：Step 2 预检主命令由 `python scripts/check_npu_env.py --build-only` 改为 `python scripts/check_npu_env.py`（完整预检），并说明完整预检同时检测运行与编译环境（`torch` / `torch_npu` / `triton-ascend` 可导入、版本下限、`torch.npu.is_available()`）；随后补充 `--build-only` 的适用场景（无 NPU 卡的纯构建环境，`is_available()` 报 `FAIL` 属预期时跳过 torch 系检查）。

### 修订 J2（README Step 2）：补充预检未覆盖的编译链依赖组件版本

- **评论**：`README.md:66`「这里是需要你去搞清楚剩余的一些依赖的版本，你通过新的环境去探索一下我们预检没覆盖的依赖组件的版本」（附一段构建日志：隔离 venv 中安装 build dependencies 的实际版本）。
- **探索结论**：预检脚本只检查 Python / bash / CANN 环境变量与 torch 系依赖，不检查实际编译链组件。经查源码确认版本要求：
  - `cmake` >= 3.16（`CMakeLists.txt` 第 12 行 `cmake_minimum_required`）；
  - `gcc` / `g++` >= 7.3（`install_deps.sh` `install_gcc`）；
  - `bisheng` >= 8.5（CANN toolkit 自带，`version.cmake` `set_build_dependencies(bisheng-compiler ">=8.5")`，`build.sh` 编译前强校验）；
  - `make`（CMake 默认构建后端；CI 镜像亦安装 `ninja-build`，二选一）；
  - `patch`（`install_deps.sh` `install_patch`；CI 镜像亦安装）；
  - Python 头文件（CI 镜像安装 `python3-dev`）；
  - `setuptools` >= 70.1 / `wheel` / `packaging` / `psutil`（`pyproject.toml` `[build-system] requires`，构建时自动安装；评论日志实测版本 setuptools 83.0.0、wheel 0.47.0、packaging 26.3、psutil 7.2.2）。
- **实际改法**：在预检命令后新增依赖组件版本表格（上述组件 + 版本要求 + 说明），并注明"预检不覆盖编译链上的工具链与构建依赖，这些组件缺失时会在 `pip wheel` 阶段才报错"。

---

## K. 第五轮修订（2026-08-08，PR #280 行内【review】评论 4 条）

reviewer 在 `docs/developer-guide.md`（3 条）与 `README.md`（1 条）新增 4 条【review】行内评论，已最小化修订并登记如下。

### 修订 K1（developer-guide 场景 3.1）：算子实现给出参考的目录结构

- **评论**：`docs/developer-guide.md:47`「算子实现给出参考的目录结构」。
- **实际改法**：场景 3.1 在目录作用表后新增以 `gdn` 模块下**反向算子** `chunk_bwd_dv_local` 为示例的参考目录树（`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/` 下的 `CMakeLists.txt`、`op_host/`（def / tiling / tiling_processor / op_api 的 aclnn 头与实现）、`op_kernel/`（kernel 实现与 cube / vector / common 结构体头）），以及对应 Python 调用侧文件（`_aclnn_ctypes.py` wrapper、`__init__.py` 注册、`test/test_npu_chunk_bwd_dv_local.py`），并注明新算子文件均可参考 `fla/ops/ascendc/gdn/` 下已有算子补齐。**第二轮修订（2026-08-08）**：按用户要求示例由前向算子 `chunk_fwd_o` 改为反向算子 `chunk_bwd_dv_local`。

### 修订 K2（developer-guide 场景 2）：修正"一键编包单算子"为全量 wheel 编包

- **评论**：`docs/developer-guide.md:35`「这里一键编包还是编译整包的指令，不是编译单算子的指令，确定一下单算子能否通过一键编包支持」。
- **探索结论**：`pip wheel` 底层走 `setup.py` 的 `_build_run_package()`，该函数调用 `bash build.sh --soc --pkg --vendor_name=fla_npu` **不带 `--ops`**，即一次编译全部已注册算子并打包成一个 wheel，**当前不支持只挑单个算子**。
- **实际改法**：场景 2 标题改为"一键编包（全量 wheel）"，正文明确"`pip wheel` 当前只支持全量编包、不支持单算子"，需要单算子 run 包时用场景 1 的 `bash build.sh --pkg ... --ops=<op>`；只改 Python wrapper 时可单独 `cd torch_custom/fla_npu && python3 setup.py bdist_wheel`。

### 修订 K3（developer-guide"确认 wheel 来源"方式 2）：新增 md5 比对脚本

- **评论**：`docs/developer-guide.md:120`「这里提供的方式一、方式二似乎并不能，我们能不能改造一下方式二，提供一个脚本，打印出调用到的 libcust_opapi.so 和编译出的 libcust_opapi.so，看 md5 值是否一样」。
- **实际改法**：新增脚本 `scripts/verify_libcust_opapi_md5.py`——导入 `fla_npu` 后读取 `FLA_NPU_OP_API_LIB` 得到运行时实际加载的 `libcust_opapi.so`，与编译产物（默认 `build/libcust_opapi.so`，或 `--built-lib <path>` 指定，或 `--run-package <path>` 自动提取 Makeself run 包内的同名库）比对 md5，输出两边路径 + md5 并给出 `[OK]`（一致）或 `[FAIL]`（不一致，需重装 wheel / run 包）结论。文档方式 2 改为该脚本用法，替换原"打印环境变量"的内联片段。**已实测**：对本地 `build_out/fla-npu-fla_npu_linux-aarch64.run` 提取后 md5 与 `build/libcust_opapi.so` 一致，输出 `[OK]`。
- **第二轮修订（2026-08-08，扩展 `--python`）**：按用户"Python wrapper 能否用类似方式区分"补充 Python wrapper 对比——脚本新增 `--python` 开关，把已安装的 `fla_npu/__init__.py`、`ops/ascendc/__init__.py`、`_aclnn_ctypes.py`、`_runtime.py` 与 `torch_custom/fla_npu/fla_npu/` 下源码逐个比对 md5（打包时 `.py` 为纯拷贝，源码改动会如实反映到已安装文件），任一不一致即报 `[FAIL]`。文档同步补充 `--python` 用法与"只改 wrapper / 只改 C++ 时各自用哪种模式"的说明。**已实测**：隔离 venv 安装完整 wheel 后 `--python` 全部 `[OK]`；临时修改源码后正确检出 `[FAIL]`（退出码 1），恢复后 `[OK]`（退出码 0）。

### 修订 K4（README Step 4）：恢复"测试单算子 / 算子调用方式 / 端到端验证"三节

- **评论**：`README.md:184`「这里测试单算子，算子调用方式和端到端验证先恢复PR修改前即可，然后算子调用方式涉及到旧接口指引跳转到升级接口的文档」。
- **实际改法**：README Step 4 恢复 PR 修改前的完整三节——`#### 测试单算子`（全量/单个命令 + `--op` 可选值 11 个，与 `test.sh` 逐条核对一致）、`#### 算子调用方式参考`（`fla_npu.ops.ascendc` / `.triton` 导入示例；末尾"使用旧版 `torch.ops.npu.*` / `torch_npu.ops.*` 的存量代码如何迁移"跳转 [兼容与迁移指南](migration-guide.md)）、`#### 端到端 Example/ST 验证`（`python examples/flash_gated_delta_rule.py` + CI 用例 schema 说明）；原"冒烟测试 + 可选的端到端验证"两个短小节删除。`docs/developer-guide.md` 场景 4/5 同步精简为引用 README Step 4（场景 4 只保留 `--mode dry-run` 开发调试选项，场景 5 只保留新增 CI 用例的 schema 说明），避免两份文档维护重复内容。

---

## L. 第六轮修订（2026-08-08，PR #280 行内【review】评论 2 条）

reviewer 在 `README.md` 与 `docs/developer-guide.md` 新增 2 条【review】行内评论，已最小化修订并登记如下。

### 修订 L1（README Step 4）：端到端 Example/ST 验证节精简，CI 用例细节移回开发者文档

- **评论**：`README.md:252`「端到端验证为什么新增了这么多内容，这部分不属于这一节吧，之前为什么有这么多内容，前面的commit是不是直接把这块移除了」。
- **核对结论**：K4 恢复的三节内容取自 `origin/main`（PR 修改前）原文，其中端到端节的 `example_st_cases.json` 管理、`gate_source` 预留说明本就是修改前 README 的原文，并非本次新增。但 reviewer 认为这些 CI 用例 schema 细节不属于"端到端验证"这一使用者向小节。
- **实际改法**：README Step 4 端到端节精简为仅保留核心的"一键运行 GDN 模块示例"（`python examples/flash_gated_delta_rule.py`），删除 `example_st_cases.json` 管理、case1 默认值、GVA/`Vdim` 泛化、`gate_source` 预留等 CI 用例 schema 细节，改为一句指向[开发者指南](developer-guide.md) 场景 5 的指引。这些细节在 developer-guide 场景 5 已有完整内容，信息不丢失。

### 修订 L2（developer-guide"确认 wheel 来源"）：删除方式 1 与方式 3，只保留 md5 脚本与打印确认

- **评论**：`docs/developer-guide.md:138`「方式一和方式三实际无效，可以直接去除，就保留我们的md5脚本和建议用户增加打印确定即可」。
- **实际改法**：章节重构为——章节开头简介后直接进入"比对运行时加载与最新编译产物的 md5"（原方式 2 升级为主方法，含 `--python` 用法与只改 wrapper / 只改 C++ 的适用说明）；原方式 1（核对 wheel 文件名与版本号）与方式 3（修改后强制覆盖安装）删除；方式 3 中的"临时打印标记"建议保留，改为独立小节"辅助确认：临时打印标记"。

---

## M. 第七轮修订（2026-08-08，md5 脚本新增 kernel `.o` 默认比对）

针对"只改 kernel 编译后脚本能否检出"的问题做了实测与脚本增强。

### 实测结论：只改 kernel，`libcust_opapi.so` md5 不变

在 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp` 加入真实进入二进制的逻辑改动（`ChunkFwdODispatch` 内新增 `if (tilingData->dataType == 42) return;`，运行时读取值不会被优化掉、实际不命中、不影响功能），完整 `pip wheel` 重新编译后：

- `libcust_opapi.so` md5 **不变**（`240beb28...`）——host 侧 aclnn 接口库不包含 kernel 二进制；
- `chunk_fwd_o` 的 4 个 kernel `.o`（`ChunkFwdO_*.o`，位于 `op_impl/ai_core/tbe/kernel/ascend910b/chunk_fwd_o/`）md5 **全部变化**——kernel `.o` 才是 NPU 上实际执行的二进制，其文件名哈希仅由参数结构决定、不随源码内容变化，但内容 md5 会变。

因此只改 kernel 时，原先只对比 `libcust_opapi.so` 的 md5 脚本无法检出"改动未生效"，需要额外对比 kernel `.o`。

### 修改 M1（scripts/verify_libcust_opapi_md5.py）：默认新增 kernel `.o` 全量比对

- **实际改法**：脚本新增 `_check_kernel_o()`——运行时侧推导内嵌 OPP 的 kernel 根目录（`lib` 的 `parents[2]/op_impl/ai_core/tbe/kernel`），build 侧默认取 `build/lib/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel`，两侧递归收集全部 `*.o` 按相对路径逐文件比对 md5，输出 `[DIFF]`（md5 不一致，含两侧 md5）与 `[MISSING]`（单侧存在）清单（超出 20 条折叠），任一不一致报 `[FAIL]`。**默认执行**（不要求额外参数），可用 `--no-kernel` 跳过；`--run-package` / `--built-lib` 场景若 build 侧 kernel 目录不存在则打印 `[WARN]` 跳过，不影响整体结论。
- **性能实测**：kernel `.o` 共 85 个、总大小约 10 MB，全量 md5 比对约 0.03 秒，远小于编译耗时，可放心默认开启。
- **端到端实测**：改 kernel 编译后、未重装时运行脚本 → lib 侧 `[OK]`、kernel 侧 `[FAIL]`（准确列出 `ascend910b/chunk_fwd_o/ChunkFwdO_*.o` 4 个 DIFF），退出码 1；安装新 wheel 后 → 两侧均 `[OK]`，退出码 0。

### 修改 M2（developer-guide"确认 wheel 来源"）：文档同步默认 kernel 比对

- **实际改法**：`docs/developer-guide.md` 该节改为说明脚本默认对比两部分（`libcust_opapi.so` + 全部 kernel `.o`），补充 `--no-kernel` 用法、kernel `.o` 规模与耗时、以及三类产物 md5 相互独立的注意点（只改 kernel → 仅 kernel 侧变；只改 host → 两者变；只改 wrapper → 需 `--python`）。

### 修改 M3（scripts/verify_libcust_opapi_md5.py）：不再依赖 import fla_npu / CANN 环境，可直接运行

- **问题**：原脚本在 `main()` 里 `import fla_npu` 以获取 `FLA_NPU_OP_API_LIB`。而 `import fla_npu` 会触发 `fla_npu/__init__.py` 的 `load_ascendc_opapi_libraries()`，用 `ctypes.CDLL` 真实加载 `libcust_opapi.so`，依赖 CANN 的 `libprofapi.so` / `libopapi_math.so`。未 source CANN `set_env.sh` 时 `LD_LIBRARY_PATH` 缺 CANN `lib64`，脚本直接抛 `OSError`，每次使用都要先 source，影响易用性。
- **实际改法**：脚本改为**纯文件 md5 比对，不再导入或加载 fla_npu**。新增 `_installed_fla_npu_root()`（用 `importlib.util.find_spec` 定位已安装包目录，不执行 `__init__.py`）与 `_resolve_runtime_lib()`（复刻 `fla_npu` 的 `_candidate_opp_roots` + `_resolve_vendor_dir` 查找顺序：`FLA_NPU_OPP_PATH` → 已安装包内嵌 OPP → `ASCEND_CUSTOM_OPP_PATH` → `ASCEND_OPP_PATH`，首个含 `op_api/lib/libcust_opapi.so` 的 root 胜出）。`_check_opp_lib` / `_check_kernel_o` / `_check_python_wrapper` 均改用该解析结果。
- **已实测**：`env -i` 清空全部环境变量后直接运行脚本 → lib / kernel / wrapper 全部 `[OK]`，退出码 0；`FLA_NPU_OPP_PATH` 指向外部 build 产物时正确解析到 build 侧；未安装 fla_npu 的 Python（系统 python）下也能运行并给出差异结论。
- **顺带修正**：kernel 对比 `[MISSING]` 单侧存在时"only on X side"标记方向写反，已改为 runtime 独有标 `runtime side`、build 独有标 `built side`。

### 修改 M4（scripts/verify_libcust_opapi_md5.py）：新增 `--built-kernel` 参数

- **实际改法**：kernel `.o` 的 build 侧目录原先硬编码为 `build/lib/fla_npu/opp/vendors/.../kernel`，不利于自定义构建目录与测试隔离。新增 `--built-kernel <path>` 参数显式指定 build 侧 kernel 根目录（默认仍为上述硬编码路径，行为不变）。

### 修改 M5（tests/test_verify_libcust_opapi_md5.py）：新增脚本测试用例

- **实际改法**：新增 `tests/test_verify_libcust_opapi_md5.py`，沿用仓库 `unittest` 风格（不引入 pytest），用临时目录构造最小 vendor OPP（`op_api/lib/libcust_opapi.so` + `op_impl/.../kernel/ascend910b/sample/*.o`）与最小已安装 fla_npu 包（`opp/` + wrapper `.py`），在干净子进程（仅 `PATH`/`HOME` + 测试所需变量）中运行脚本并断言退出码与输出。覆盖以下用例：
  - 默认完整对比（lib + kernel）一致 → `[OK]`，退出码 0；
  - 只改 kernel（lib 相同、kernel `.o` 内容不同）→ lib `[OK]`、kernel `[FAIL]`，退出码 1；
  - 只改 lib → `[FAIL]`，退出码 1；
  - `--no-kernel` 跳过 kernel 对比 → 退出码 0；
  - kernel `.o` 仅 build 侧存在 → `[MISSING] ... (only on built side)`，退出码 1；
  - `--python` wrapper 一致 → `[OK]`；篡改 installed `__init__.py` → `[FAIL]`；
  - `FLA_NPU_OPP_PATH` 优先于已安装包内嵌 OPP；
  - 无可解析 OPP → `[FAIL] runtime libcust_opapi.so not found`，退出码 1；
  - `--help` 列出全部参数。
- **测试结果**：10 个用例全部通过（`python -m unittest tests.test_verify_libcust_opapi_md5 -v`）。测试过程中发现并修复了测试自身的两个问题：`--built-lib` 路径拼错（少了 `vendors/fla_npu_transformer` 中间层）、wrapper 一致用例需从真实源码复制文件而非空内容。
- **文档同步**：`docs/developer-guide.md` 该节补充 `--built-kernel` 用法与测试入口。

---

## N. 第八轮修订（2026-08-11）

本轮完成两项改动：`check_npu_env.py` 预检补充 cmake / setuptools 版本检测；README 方式 B 从主文档迁移到开发者指南。

### 修改 N1（scripts/check_npu_env.py）：预检补充 cmake / setuptools 版本检测

- **背景**：PR #280 评论要求"把预检没覆盖的工具链依赖也加上，分析怎么判断这些依赖"（README Step 2 表格列出 `cmake` / `gcc` / `bisheng` / `make` / `patch` / Python 头 / `setuptools` 等，但 `check_npu_env.py` 此前不检测）。本轮先落地 cmake 与 setuptools 两项，用实测确定最小版本。
- **实测确定最小版本**：
  - **cmake**：`CMakeLists.txt` 声明 `cmake_minimum_required(VERSION 3.16)`，低于 3.16 的版本会被 CMake 自身拒绝启动；`install_deps.sh` 亦以 `req_ver="3.16.0"` 检查。项目用到的 CMake 特性（`cmake_parse_arguments` 3.5+、`target_link_options` 3.13+）均不高于 3.16。**结论：cmake >= 3.16 均可支持。**
  - **setuptools**：`pyproject.toml` 声明 `setuptools>=70.1`；实测 setuptools 69.5.1 **没有** `setuptools.command.bdist_wheel`（`setup.py` 回退链第二步依赖它），80.9.0 有；setuptools 70.x 起在 `setuptools.command.bdist_wheel` 内置 bdist_wheel（不再依赖独立 `wheel` 包）。**结论：setuptools >= 70.1 均可支持。**
- **实际改法**：新增 `_check_cmake_version()`（`shutil.which` 定位 cmake，`cmake --version` 解析版本，`>=3.16` 校验，缺失/无法解析版本时 `[FAIL]`）与 `_check_setuptools_version()`（`importlib.metadata.version` 取版本，`>=70.1` 校验）。两者在 CANN 环境检查之后、torch 系检查之前执行。
- **已实测**：
  - cmake 4.3.1 + setuptools 80.9.0 → 均 `[OK]`；
  - setuptools 69.5.1 → `[FAIL] setuptools>=70.1 is required, got 69.5.1`；
  - 模拟 cmake 3.10.2 → `[FAIL] cmake>=3.16 is required`；模拟 3.16.0 → `[OK]`（边界正确）；
  - `shutil.which` 找不到 cmake → `[FAIL] cmake not found`。
- **文档同步**：README Step 2 工具链表格保持不变（`cmake >= 3.16`、`setuptools>=70.1` 与脚本一致）；表格下方预检说明补一句"cmake / setuptools 已纳入预检，其余组件缺失时仍在 `pip wheel` 阶段报错"。

### 修改 N2（README.md）：方式 B 从主 README 迁移到开发者指南

- **背景**：用户要求"README 的方式 B 不要存在于主 README，应引导用户使用最新方式（一键编包全量 wheel + 安装），可移动到其他合适位置作必要时的参考"。
- **实际改法**：
  - Step 2：删除"方式 B：【备选】单独编译算子 run 包和 Python wheel"整节（含 `bash build.sh --pkg ... --ops=` 与 `cd torch_custom/fla_npu && python3 setup.py bdist_wheel`），改为一行引用块"需要单独编译一个或多个算子 run 包的开发者场景，见开发者指南场景 1"；"方式 A：【推荐】源码一键编译并生成 wheel"标题改为"【推荐】源码一键编译并生成 wheel"，"方式 A 编译可用环境变量"改为"编译可用环境变量"。
  - Step 3：删除"方式 A 产物安装" / "方式 B 产物安装"两个小节标题与 run 包安装细节（安装器算子状态 `WARNING` / `NOTICE` / `OK` 说明、`./build_out/fla-npu-*.run --install` / `--full`、run 包覆盖后重写 `set_env.bash` 与 `RECORD`），保留 wheel 安装主流程与通用 runtime 加载说明（`import fla_npu` 定位 OPP、`fla_npu.ops.ascendc` 查找顺序），末尾加引用块"已安装完整 wheel 后如需用单算子 run 包快速替换部分算子产物（含安装器算子状态说明），见开发者指南场景 1"。
  - Step 4：验证命令说明由"安装后两种方式均可用以下命令验证"改为"安装后可用以下命令验证"。
- **同步迁移**：`docs/developer-guide.md` 场景 1 补充"安装 run 包"小节，承接原 README 方式 B 的完整安装细节（安装器算子状态说明、`--install` / `--full` 命令、Python wrapper wheel 重装、`set_env.bash` / `RECORD` 行为），并加一句"常规使用者推荐直接用根 README Step 2 / Step 3 的一键编包 + wheel 安装主流程；本场景仅在需要快速替换单个算子产物时使用"。
- **引用一致性检查**：README / developer-guide / migration-guide 中"方式 A / 方式 B"字样已全部清除或改写；migration-guide 的 Step 引用与 run 包描述不受影响。

### 修改 N3（scripts/check_npu_env.py + README 工具链表格）：实测并纳入 gcc / make / bisheng 检查，修正表格不准确项

- **背景**：用户指出 README 工具链表格（cmake / gcc / bisheng / make / patch / Python 头 / setuptools）是此前基于评论总结的，未逐项实测，要求实测真实依赖并纳入预检。
- **实测结论（逐项核实代码调用链）**：
  - `gcc` / `g++`：真实依赖。`install_deps.sh` 以 `req_ver="7.3.0"` 检查；build.sh 多处用 `g++` 做 host 侧编译。**最小版本 >= 7.3**。
  - `make`：真实依赖。CMake 默认生成器为 `Unix Makefiles`（`cmake/custom_build.cmake` 的 `CPACK_CMAKE_GENERATOR "Unix Makefiles"`），build.sh 用 `make clean`；ninja 为等价替代。
  - `bisheng`：真实依赖（kernel 编译）。`build.sh` 用 `which bisheng` 定位、缺失即报错退出；但 `bisheng --version` 输出的是 clang 版本（实测 15.0.5），**无法**解析出 CANN 组件版本，故只做存在性检查，版本要求 `>=8.5` 是 CANN 组件版本（`version.cmake` 声明）而非 bisheng 自身可判断的版本。
  - `patch`：**并非硬依赖**。全仓库主构建流程（setup.py / build.sh / pip wheel）均未调用外部 `patch` 命令；`scripts/package/common/sh/install_common_parser.sh` 中的 `install_patch` 是 shell 函数名，非 `patch` 命令。**已从 README 表格移除。**
  - Python 头文件：仅 `FLA_NPU_BUILD_LEGACY_EXTENSION=1` 编译 legacy C++ 扩展时需要（`setup.py` 的 `_build_torch_extension_inplace` 与 `torch_custom/fla_npu/setup.py` 的 `_setup_legacy_extension`），默认 wheel 构建不需要。**表格说明改为"仅 legacy 构建需要"。**
  - `setuptools` / `wheel` / `packaging` / `psutil`：均为 pyproject build-system 声明。README 推荐 `pip wheel --no-build-isolation`，故这些包需本机已装。其中 setuptools 有版本下限 70.1（上一轮已纳入），wheel / packaging / psutil 无版本下限，仅需存在。
  - 附注：PR 评论附带 CI 构建日志显示隔离环境实际安装 `cmake<4,>=3.16` 与 `patch-ng==1.19.1`——前者是 CI 对 CMake 的安全版本上界（项目本身 `cmake_minimum_required` 无上界），后者是 Python 库（patch-ng，非外部 `patch` 命令），均非项目声明的硬约束。
- **实际改法**：`check_npu_env.py` 新增 `_check_gcc_version()`（gcc / g++ 分别 `--version` 解析 `x.y.z` 并 `>=7.3` 校验）、`_check_make_exists()`（make 或 ninja 任一存在即可）、`_check_bisheng_exists()`（仅存在性，缺失时提示随 CANN 安装并 source 环境）、`_check_build_system_deps()`（importlib 导入 wheel / packaging / psutil 确认存在）、`_tool_version()`（通用 `tool --version` 正则解析帮助函数）。cmake 检查复用了 `_tool_version` 的解析思路。均在 CANN 环境检查之后执行。
- **已实测**：
  - 本机（cmake 4.3.1、gcc 11.4.0、make 4.3、wheel 0.45.1、packaging 25.0、psutil 7.1.3、无 bisheng）→ cmake / gcc / g++ / make / setuptools / wheel / packaging / psutil `[OK]`，bisheng `[FAIL]`（预期，未 source CANN）；
  - 模拟 gcc/g++ 7.2.0 → `[FAIL] gcc>=7.3 is required`；模拟 7.3.0 → `[OK]`（边界正确）；
  - make / ninja 均缺失 → `[FAIL] make not found`；模拟 bisheng 存在 → `[OK]`。
- **文档同步**：README Step 2 工具链表格移除 `patch` 行，`Python 头文件` 改为"仅 legacy 构建需要"，`bisheng` 版本要求改为"随 CANN（无独立版本判断）"，`make` 说明补"CMake 默认 Unix Makefiles 生成器后端（ninja 亦可）"，build-system 行说明补"`--no-build-isolation` 构建时需本机已装"；预检说明改为"覆盖 cmake、gcc/g++、setuptools 版本要求，make / bisheng 存在性检查，wheel / packaging / psutil 导入检查"。

## O. 第九轮修订（2026-08-11）：处理 PR #280 新增 8 条 review 评论

本轮按 PR #280 新增的 8 条行内 review 评论逐条判断合理性后做最小化改动。

### 修改 O1（torch_custom/fla_npu/test/test.sh + README 测试节）：修复 PYTHONPATH 遮蔽已安装 wheel OPP

- **评论（3757854004）**：`test.sh` 第 21 行无条件 `export PYTHONPATH=<源码树>:...`，`import fla_npu` 会命中源码树 `torch_custom/fla_npu/fla_npu`（其 OPP 目录只有 `README.txt` 骨架，无 `libcust_opapi.so`），遮蔽已安装 wheel 的内嵌 OPP，导致按 README 构建安装后首个验证命令失败。**合理。**
- **实际改法**：test.sh 改为仅当源码树 `fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so` 存在时才 prepend 源码路径；只安装 wheel 的用户直接运行 test.sh 即用已安装 OPP。README 测试节补一句行为说明。
- **附带说明**：部分用例（`chunk_bwd_dv_local` / `prepare_wy_repr_bwd_full` / `recompute_w_u_fwd`）精度比对依赖内部 `ct` 模块（不随 CANN / requirements 提供），全量测试时 `ModuleNotFoundError` 属预期；README 注明单算子冒烟建议选 `gdn_fwd_o` 等不依赖 `ct` 的用例。

### 修改 O2（requirements.txt）：补充 ml_dtypes

- **评论（3757854192）**：`test_fwd_o.py` / `test_fwd_h.py` 顶层 `import ml_dtypes`，但 requirements.txt 未包含；按 README 安装后跑测试必失败（实测补装 ml_dtypes==0.5.4 后 PASS）。**合理。**
- **实际改法**：requirements.txt 追加 `ml_dtypes`。

### 修改 O3（setup.py）：根 wheel 补充 fla_npu.ops.triton.triton_core 子包

- **评论（3757854293）**：根 `setup.py` 的 `find_packages(where=torch_custom/fla_npu)` 不会扫描位于仓库根 `fla/ops/triton/triton_core` 的源码，而 `torch_custom/fla_npu/setup.py` 已通过 `TRITON_CORE_PACKAGE` / `package_dir` 映射包含它，根 setup.py 未同步，导致按 README 构建的 wheel 缺该子包，`from fla_npu.ops.triton import ...` 直接 `ModuleNotFoundError`。**合理。**
- **实际改法**：根 setup.py 新增 `TRITON_CORE_PACKAGE` / `TRITON_CORE_SOURCE` 常量与 `_find_packages()` / `_find_package_dir()` 帮助函数，setup() 的 packages / package_dir 改用它们（逻辑与 torch_custom/fla_npu/setup.py 一致）。已验证 `_find_packages()` 返回含 `fla_npu.ops.triton.triton_core` 且映射正确；`tests/test_wheel_environment.py` 8 个用例全部通过。

### 修改 O4（README Step 1）：ops 包安装后增加核对手段

- **评论（3757854436）**：实测 A2(910B3) 误装 `Ascend-cann-910-ops`（旧 910 芯片包）后原生 matmul 全挂（错误码 561103），排障成本高；README 虽给出正确命名示例，但无安装后核对手段。**合理。**
- **实际改法**：Step 1 安装命令后增加核对示例——`grep package_name $ASCEND_OPP_PATH/../aarch64-linux/ascend_ops_install.info` 应显示对应芯片的 ops 包名。

### 修改 O5（scripts/check_npu_env.py + README）：triton-ascend 按 CANN 版本分层

- **评论（3757854541）**：预检只要求 `triton-ascend>=3.2.0`，但实测 3.2.0 在 CANN 9.1.0 上 JIT 编译 `npu_utils.cpp` 失败（`rt.h` 缺 `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`）；CI `ci/9.0.0/Dockerfile`（CANN 9.1.0-beta.1）实际用 3.2.1。**合理。**
- **实际改法**：新增 `MIN_TRITON_ASCEND_CANN9 = "3.2.1"` 与 `_check_triton_ascend_cann_compat()`，解析 `_detect_cann_version()` 返回的 CANN major 版本：9.x 要求 triton-ascend >= 3.2.1，8.x 维持 >= 3.2.0。README Step 2 预检说明补版本匹配提示。已实测：CANN 9.1.0 + 3.2.0 FAIL、+ 3.2.1 PASS、CANN 8.3 + 3.2.0 PASS。

### 修改 O6（scripts/check_npu_env.py）：_detect_cann_version 优先读 OPP、过滤驱动版本

- **评论（3757861181）**：`_detect_cann_version()` 在部分安装布局下读到驱动版本文件而非 CANN 版本（实测输出 `version=25.5.1`，实际 CANN 9.1.0）。**合理。**
- **实际改法**：重写 `_detect_cann_version()`：优先读 `ASCEND_OPP_PATH` 下的 `version.info`（权威 CANN 版本，如 `Version=8.3.0.1.200` / `Version=9.1.0-beta.1`），过滤含 `driver` 的行，`ASCEND_HOME_PATH` 作为兜底。已用本机真实 CANN 8.3.RC1 / 9.1.0-beta.1 安装目录实测正确。

### 修改 O7（README + developer-guide）：工具链依赖表移出主 README

- **评论（3757865945）**：依赖表格不应放在主 README，应放开发者文档。**合理。**
- **实际改法**：README Step 2 表格整体移除，压缩为一句指引（预检覆盖项 + 指向开发者指南）；完整表格移入 `docs/developer-guide.md` 场景 2 新增的"工具链依赖表"小节。

### 评论 O8（结论后续被推翻）：FLA_NPU_OPS 应保留

- **评论（3757737653）**：建议用 `FLA_NPU_OPS` 环境变量控制一键编包时编译单算子并实测补充到文档。当时判断**不成立**——`FLA_NPU_OPS` 在 PR #274 从 setup.py 移除（`tests/test_wheel_environment.py` 有守卫断言其不得出现），`pip wheel` 只支持全量编包，单算子场景走 developer-guide 场景 1 的 `bash build.sh --pkg ... --ops=<op>` run 包方式，并已在评论下回复说明。
- **后续用户结论（覆盖 O8）**：PR #274 移除 `FLA_NPU_OPS` 属于**错误移除，应保留**。已据此恢复：`setup.py` `_build_run_package()` 恢复读取 `FLA_NPU_OPS` 并透传 `build.sh --ops=<op>`；README 环境变量表加回 `FLA_NPU_OPS` 行；developer-guide 场景 2 补充 `FLA_NPU_OPS` 单算子 wheel 用法；`tests/test_wheel_environment.py` 移除 `assertNotIn("FLA_NPU_OPS")` 并新增 `test_package_build_supports_single_op_filter` 断言其存在；migration-guide 差异表同步更新。

## P. 第十轮修订（2026-08-18）：处理 issue #63 + PR #280 新增 3 条 review 评论

### 修订 P1（issue #63）：check_npu_env.py 预检新增 patch

- **issue（#63）**：编译安装缺少 `cmake`、`patch` 等系统命令时 README 无提示，构建到第三方源码阶段才报错。8/17 补充评论明确要求：预检 `cmake`（>=3.16）与 `patch`，缺失立即提示，保持 pyproject.toml / 预检 / README 一致。
- **判断**：`cmake>=3.16` 已在预检覆盖（此前修订）；**patch 未覆盖**——构建在 `cmake/third_party/abseil-cpp.cmake:55` 与 `ascend_protobuf.cmake:57` 的 `PATCH_COMMAND` 中调用系统 `patch`，缺失只在中途暴露。
- **实际改法**：`check_npu_env.py` 新增 `_check_patch_exists()`（存在性检查，无版本下限，缺失时 `[FAIL]` 并给出 `apt-get install -y patch` 提示）；README Step 2 预检描述与 developer-guide 工具链依赖表同步加 `patch` 行。已实测：本机 `[OK] patch`；空 PATH 模拟缺失时 `[FAIL]` 正确输出。
- **未改**：`pyproject.toml` 不声明 cmake/patch——`[build-system] requires` 只负责 pip 隔离构建的 Python 包，系统命令应由预检 + 文档覆盖（`--no-build-isolation` 也无需它）。

### 修订 P2（migration-guide）：强调新接口稳定、旧方式简单提及

- **评论（3801853040）**：`docs/migration-guide.md:37`「显示要求用户使用最新接口，承诺最新接口不会变动，旧的使用方式简单提及即可」。
- **实际改法**：背景表格改为"状态"列（旧方式已停止演进 / 新接口推荐且稳定）；"迁移期临时兼容"章节开头新增承诺——`fla_npu.ops.ascendc` 是稳定入口、后续版本不会变动；legacy 路径描述精简（去掉冗余的生成文件清单，保留 `install_torch_npu_ops_compat()` 的 `AttributeError` 注意与 legacy 构建命令）。

### 修订 P3（developer-guide）：新增"调用链路与场景选择"

- **评论（3802064080）**：`docs/developer-guide.md:11`「增加调用链路图（让用户明白什么时候改什么）、各场景适配的情况（什么时候用哪个场景）」。
- **实际改法**：developer-guide 开头新增"调用链路与场景选择"章节：ASCII 链路图（源码修改 → 构建产物 → 验证 → 确认生效，标注各场景）+ 场景选择表（你想做什么 → 用哪个场景 → 产物）；目录同步加锚点。README 现有"场景 2 的工具链依赖表"等引用保持有效。

### 修订 P4（developer-guide 场景 2）：标题与 FLA_NPU_OPS 对齐

- **评论（3801871866）**：`docs/developer-guide.md:62`「一键编包支持单算子 FLA_NPU_OPS 相关内容也要放在这」。
- **判断**：`FLA_NPU_OPS` 单算子 wheel 用法在恢复 `FLA_NPU_OPS` 时已补入场景 2；评论希望该能力在场景 2 更明确。
- **实际改法**：场景 2 标题由"一键编包（全量 wheel）"改为"一键编包（wheel）"，引导句明确"默认全量，或用 `FLA_NPU_OPS` 只编指定算子"，与已恢复的 `FLA_NPU_OPS` 单算子 wheel 用法保持一致。
